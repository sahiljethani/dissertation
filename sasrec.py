import argparse
from recbole.quick_start import run_recbole

def sasrec(dataset):

    parameter_dict = {
        'train_neg_sample_args': None,
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 2048,
        'learning_rate': 0.001,
        'embedding_size': 64,
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 2,
        'dropout_prob': 0.2,
        'loss_type': 'CE',
        'metrics': ["Hit", "NDCG"],
        'topk': [10, 20],
        'valid_metric': 'NDCG@10',
        'data_path': 'processed',
        'dataset': dataset,
        'field_separator': "\t",
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'ITEM_SEQ_FIELD': 'item_id_list',
        'load_col': {'inter': ['user_id', 'item_id_list','item_id', 'timestamp']}
    }

    # Run the RecBole model
    result = run_recbole(model='SASRec', config_dict=parameter_dict)
    
    # Print the final test results
    print("Final Test Results:")
    print(result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='All_Beauty', help='Name of the dataset')
    args = parser.parse_args()

    print(args)
    print('Running SASRec on dataset:', args.dataset)

    sasrec(args.dataset)

