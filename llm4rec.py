import argparse
import torch
import faiss
import json
from evaluate import metrics_10
import os

def run(domain,path):

    domain_path=os.path.join(path, domain)

    #loading pre-trained embeddings
    item_profile_embeddings = torch.load(os.path.join(domain_path,'item_profile.pth'))
    valid_user_embeddings = torch.load(os.path.join(domain_path,'valid_user_behavior.pth'))
    test_user_embeddings = torch.load(os.path.join(domain_path,'test_user_behavior.pth'))
    

    #loading data maps
    with open(f'{domain_path}/{domain}.data_maps', 'r') as f:
        data_maps = json.load(f) 

    #getting top 50 items for test and valid
    index = faiss.IndexFlatIP(768)
    index.add(item_profile_embeddings.cpu().detach().numpy())

    distances_test, indices_test = index.search(test_user_embeddings.cpu().detach().numpy(), 50)

    predictions_test = []
    for (d, idx) in (zip(distances_test, indices_test)):
        top_50_items = [data_maps['id2item'][i] for i in idx]
        predictions_test.append(top_50_items)

    distances_valid, indices_valid = index.search(valid_user_embeddings.cpu().detach().numpy(), 50)

    predictions_valid = []
    for (d, idx) in (zip(distances_valid, indices_valid)):
        top_50_items = [data_maps['id2item'][i] for i in idx]
        predictions_valid.append(top_50_items)

    #target for train and valid
    with open(f'{domain_path}/test_target.txt', 'r') as f:
        test_target = f.readlines()
    test_target = [line.strip() for line in test_target]

    with open(f'{domain_path}/valid_target.txt', 'r') as f:
        valid_target = f.readlines()
    valid_target = [line.strip() for line in valid_target]

    
    #evaluat metric for top 50 items
    k_list=[5,10,50]
    for k in k_list:
        print(f'For test set')
        ndcg, hr = metrics_10(test_target, predictions_test,k)
        print(f'NDCG@{k}: {ndcg:.4f}')
        print(f'HR@{k}: {hr:.4f}')

        print(f'For valid set')
        ndcg, hr = metrics_10(valid_target, predictions_valid,k)
        print(f'NDCG@{k}: {ndcg:.4f}')
        print(f'HR@{k}: {hr:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='All_Beauty', help='dataset name')
    parser.add_argument('--path', type=str, default='processed', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    print(f'LLM for for {args.domain}')

    run(args.domain,args.path)
