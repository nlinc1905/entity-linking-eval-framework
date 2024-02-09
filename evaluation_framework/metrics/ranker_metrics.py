"""
These functions were adapted from the ml_metrics library:
https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
"""
import numpy as np


def ap_at_k(y_true: list, y_pred: list, k=10) -> float:
    """
    Calculates the average precision at k.

    :param y_true: A list of query results where the results are the ground
        truth labels (the documents that actually are relevant to the query),
        e.g. [[id1, id2, id3], [id1, id2, id3], ...]
        The order of this list does not matter.
    :param y_pred: A list of query results where the results are predicted
        (the documents that a model thinks are relevant to the query),
        e.g. [[id1, id2, id3], [id1, id2, id3], ...]
        The items in the sublist are assumed to be ordered the best way
        that the model sees fit.
    :param k: Max number of predicted ranks to evaluate.
    """
    # filter to top k (assumes y_pred is ordered)
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    score = 0.0
    num_hits = 0.0

    # count how many predictions in the top k are in y_true
    # weight the hits according to the order they are found
    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not y_true:
        return 0.0

    final_score = score / min(len(y_true), k)
    return final_score


def map_at_k(y_true: list, y_pred: list, k: int = 10) -> float:
    """
    Calculates the mean average precision at k.

    :param y_true: Ground truth labels as a list of lists, e.g.
        [[1,2,3], [1,2,3], ...]  where each sublist represents a query
        and the ranks in the sublist are the true ranks for the results
        for that query
    :param y_pred: Predicted ranks as a list of lists, e.g.
        [[1,2,3], [1,2,3], ...]  where each sublist represents a query
        and the ranks in the sublist are the predicted ranks for the results
        for that query
    :param k: Max number of predicted ranks to evaluate.
    """
    return np.mean([ap_at_k(y_true=a, y_pred=p, k=k) for a, p in zip(y_true, y_pred)])
