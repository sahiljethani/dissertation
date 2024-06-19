import os
import re
import html
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='All_Beauty')
    parser.add_argument('--max_his_len', type=int, default=50)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='processed/')
    parser.add_argument('--plm', type=str, default='sentence-transformers/sentence-t5-base')
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_items_wo_metadata(example, item2meta):
    if example['parent_asin'] not in item2meta:
        example['history'] = ''
    history = example['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    example['history'] = ' '.join(filtered_history)
    return example


def truncate_history(example, max_his_len):
    example['history'] = ' '.join(example['history'].split(' ')[-max_his_len:])
    return example


def remap_id(datasets):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id, item_id, history in zip(dataset['user_id'], dataset['parent_asin'], dataset['history']):
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)
            if item_id not in item2id:
                item2id[item_id] = len(id2item)
                id2item.append(item_id)
            items_in_history = history.split(' ')
            for item in items_in_history:
                if item not in item2id:
                    item2id[item] = len(id2item)
                    id2item.append(item)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}
    return data_maps


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '


def clean_metadata(example):
    meta_text = ''
    features_needed = ['title', 'features', 'categories', 'description']
    for feature in features_needed:
        meta_text += feature_process(example[feature])
    example['cleaned_metadata'] = meta_text
    return example


def process_meta(args):
    domain = args.domain

    meta_dataset = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        f'raw_meta_{domain}',
        split='full',
        trust_remote_code=True
    )

    meta_dataset = meta_dataset.map(
        clean_metadata,
        num_proc=args.n_workers
    )

    item2meta = {}
    for parent_asin, cleaned_metadata in zip(meta_dataset['parent_asin'], meta_dataset['cleaned_metadata']):
        item2meta[parent_asin] = cleaned_metadata

    return item2meta


if __name__ == '__main__':
    args = parse_args()
    
    #if cude is available, use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    '''
    Process interaction sequences and item metadata
    '''
    print(f"Processing {args.domain}...")

    datasets = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"0core_timestamp_w_his_{args.domain}",
        trust_remote_code=True
    )
    item2meta = process_meta(args)

    truncated_datasets = {}
    output_dir = os.path.join(args.output_dir, args.domain)
    check_path(output_dir)
    for split in ['train', 'valid', 'test']:
        # Remove lines w/ empty history
        filtered_dataset = datasets[split].map(
            lambda t: filter_items_wo_metadata(t, item2meta),
            num_proc=args.n_workers
        )
        filtered_dataset = filtered_dataset.filter(lambda t: len(t['history']) > 0)
        # Truncate history
        truncated_dataset = filtered_dataset.map(
            lambda t: truncate_history(t, args.max_his_len),
            num_proc=args.n_workers
        )
        truncated_datasets[split] = truncated_dataset

        output_path = os.path.join(output_dir, f'{args.domain}.{split}.inter')
        with open(output_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for user_id, history, parent_asin in zip(
                truncated_dataset['user_id'],
                truncated_dataset['history'],
                truncated_dataset['parent_asin']
            ):
                f.write(f"{user_id}\t{history}\t{parent_asin}\n")

    print("Interaction sequences and item metadata processed.")

    '''
    Remap IDs
    '''
    print("Remapping IDs...")
    data_maps = remap_id(truncated_datasets)
    id2meta = {0: '[PAD]'}
    for item in item2meta:
        if item not in data_maps['item2id']:
            continue
        item_id = data_maps['item2id'][item]
        id2meta[item_id] = item2meta[item]
    data_maps['id2meta'] = id2meta
    output_path = os.path.join(output_dir, f'{args.domain}.data_maps')
    with open(output_path, 'w') as f:
        json.dump(data_maps, f)

    print("IDs remapped.")

    '''
    Generate item features
    '''
    print("Generating item features...")

    device = torch.device(device)
    
    # model = AutoModel.from_pretrained(args.plm).to(device)
    model = SentenceTransformer(args.plm, device=device)


    sorted_text = []    # 1-base, sorted_text[0] -> item_id=1
    for i in range(1, len(data_maps['item2id'])):
        sorted_text.append(data_maps['id2meta'][i])

    item_profile=model.encode(sorted_text, convert_to_tensor=True,device=str(device))
    torch.save(item_profile, os.path.join(output_dir, 'item_profile.pth'))


    with open(os.path.join(output_dir, 'item_profile.txt'), 'w') as f:
        for line in sorted_text:
            f.write(f"{line}\n")

    print("Item features generated.")

    '''
    Generate User features
    '''
    print("Generating User features...")
    user_behavior = {'train': [], 'valid': [], 'test': []}
    target = {'train': [], 'valid': [], 'test': []}

    for split in ['train', 'valid', 'test']:
        for user_id, history, parent_asin in zip(truncated_datasets[split]['user_id'], truncated_datasets[split]['history'], truncated_datasets[split]['parent_asin']):
            item_titles = []
            for item_id in history.split(' '):
                item_title = item2meta[item_id].split('.')[0]  # Extract the item title from metadata
                item_titles.append(item_title)
            user_behavior[split].append(f"User bought items: {', '.join(item_titles)}")
            target[split].append(parent_asin)
        
        user_behavior[split] = model.encode(user_behavior[split], convert_to_tensor=True,device=str(device))
        target[split] = model.encode(target[split], convert_to_tensor=True,device=str(device))
        torch.save(user_behavior[split], os.path.join(output_dir, f'{split}_user_behavior.pth'))
        torch.save(target[split], os.path.join(output_dir, f'{split}_target.pth'))

        behavior_output_path = os.path.join(output_dir, f'{split}_user_behavior.txt')
        target_output_path = os.path.join(output_dir, f'{split}_target.txt')

        with open(behavior_output_path, 'w') as f:
            for line in user_behavior[split]:
                f.write(f"{line}\n")
        
        with open(target_output_path, 'w') as f:
            for line in target[split]:
                f.write(f"{line}\n")

    print("User features generated.")


    # '''
    # Statistics
    # '''
    # print(f"#Users: {len(data_maps['user2id']) - 1}")
    # print(f"#Items: {len(data_maps['item2id']) - 1}")
    # n_interactions = {}
    # for split in ['train', 'valid', 'test']:
    #     n_interactions[split] = len(truncated_datasets[split])
    #     for history in truncated_datasets[split]['history']:
    #         if len(history.split(' ')) == 1:
    #             n_interactions[split] += 1
    # print(f"#Interaction in total: {sum(n_interactions.values())}")
    # print(n_interactions)
    # avg_his_length = 0
    # for split in ['train', 'valid', 'test']:
    #     avg_his_length += sum([len(_.split(' ')) for _ in truncated_datasets[split]['history']])
    # avg_his_length /= sum([len(truncated_datasets[split]) for split in ['train', 'valid', 'test']])
    # print(f"Average history length: {avg_his_length}")
    # print(f"Average character length of metadata: {np.mean([len(_) for _ in sorted_text])}")
