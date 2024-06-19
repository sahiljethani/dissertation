import numpy as np
import math


def ndcg_at_k(relevance, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in relevance:
        rel = row[:k]
        one_ndcg = 0.0
        for i in range(len(rel)):
            one_ndcg += rel[i] / math.log(i+2,2)
        ndcg += one_ndcg
    return ndcg
        
    
def hit_at_k(relevance, k):
    correct = 0.0
    for row in relevance:
        rel = row[:k]
        if sum(rel) > 0:
            correct += 1
    return correct
        

def metrics_10(target, predictions,k):
    ndcg_scores = []
    hit_ratios = []
    for i, gt_item in enumerate(target):
        relevance_scores = [[1 if item == gt_item else 0 for item in predictions[i]]]
        ndcg = ndcg_at_k(relevance_scores, k)
        ndcg_scores.append(ndcg)
        hr = hit_at_k(relevance_scores, k)
        hit_ratios.append(hr)
    average_ndcg = np.mean(ndcg_scores)
    average_hr = np.mean(hit_ratios)
    return average_ndcg, average_hr