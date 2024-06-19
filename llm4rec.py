import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
import sentence_transformers
import numpy as np
import logging
from tqdm import tqdm
import faiss
import json
from evaluate import metrics_10


def run(domain):

    path=f'dataset/processed/{domain}/'

    #loading pre-trained embeddings
    item_profile_embeddings = torch.load(f'{path}/item_profile_embeddings.pth')
    valid_user_embeddings = torch.load(f'{path}/valid_user_embeddings.pth')
    test_user_embeddings = torch.load(f'{path}/test_user_embeddings.pth')
    

    #loading data maps
    with open(f'{path}/{domain}.data_maps', 'r') as f:
        data_maps = json.load(f) 

    #getting top 50 items for test and valid
    index = faiss.IndexFlatIP(768)
    index.add(item_profile_embeddings)

    distances_test, indices_test = index.search(test_user_embeddings, 50)

    predictions_test = []
    for (d, idx) in (zip(distances_test, indices_test)):
        top_50_items = [data_maps['id2item'][i] for i in idx]
        predictions_test.append(top_50_items)

    distances_valid, indices_valid = index.search(valid_user_embeddings, 50)

    predictions_valid = []
    for (d, idx) in (zip(distances_valid, indices_valid)):
        top_50_items = [data_maps['id2item'][i] for i in idx]
        predictions_valid.append(top_50_items)

    #target for train and valid
    with open(f'{path}/test_target.txt', 'r') as f:
        test_target = f.readlines()
    test_target = [line.strip() for line in test_target]

    with open(f'{path}/valid_target.txt', 'r') as f:
        valid_target = f.readlines()
    valid_target = [line.strip() for line in valid_target]

    
    #evaluat metric for top 50 items
    k_list=[5,10,50]
    for k in k_list:
        print(f'For test set')
        ndcg, hr = metrics_10(test_target, top_50_items,k)
        print(f'NDCG@{k}: {ndcg:.4f}')
        print(f'HR@{k}: {hr:.4f}')

        print(f'For valid set')
        ndcg, hr = metrics_10(valid_target, top_50_items,k)
        print(f'NDCG@{k}: {ndcg:.4f}')
        print(f'HR@{k}: {hr:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='All_Beauty', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    print(f'LLM for for {args.d}')

    run(args.d)
