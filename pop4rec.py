import argparse
import pandas as pd
from evaluate import metrics_10


def popularity(domain):

    #load data
    train_dataset_path=f'dataset/processed/{domain}/{domain}.train.inter'
    test_dataset_path=f'dataset/processed/{domain}/{domain}.test.inter'

    #load train and test data
    train_data = pd.read_csv(train_dataset_path, sep='\t', header=None)
    train_data.columns = ['user_id', 'item_seq','item_id']
    train_data=train_data[1:]

    test_data = pd.read_csv(test_dataset_path, sep='\t', header=None)
    test_data.columns = ['user_id', 'item_seq','item_id']
    test_data=test_data[1:]

    #list of target items
    test_target=list(test_data['item_id'])

    #top 50 items
    top_50_items=train_data['item_id'].value_counts().head(50).index.tolist()

    #evaluat metric for top 50 items
    k_list=[5,10,50]
    for k in k_list:
        ndcg, hr = metrics_10(test_target, top_50_items,k)
        print(f'NDCG@{k}: {ndcg:.4f}')
        print(f'HR@{k}: {hr:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='All_Beauty', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    print(f'Popularity for {args.d}')

    popularity(args.d)
