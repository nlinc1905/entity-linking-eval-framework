import pandas as pd


def get_true_positives(true: pd.MultiIndex, predictions: pd.MultiIndex):
    """
    Returns true positive links.

    :param true: Ground truth links
    :param predictions: Predicted links
    """
    return true.intersection(predictions)


def get_false_positives(true: pd.MultiIndex, predictions: pd.MultiIndex):
    """
    Returns false positive links.

    :param true: Ground truth links
    :param predictions: Predicted links
    """
    return predictions.difference(true)


def get_true_negatives(candidate_links: pd.MultiIndex, true: pd.MultiIndex, predictions: pd.MultiIndex):
    """
    Returns true negative links.

    :param candidate_links: All pairs of nodes that could be compared, regardless of whether they are
        truly linked.
    :param true: Ground truth links
    :param predictions: Predicted links
    """
    not_a_link = candidate_links.difference(true)
    not_predicted = candidate_links.difference(predictions)
    return not_a_link.intersection(not_predicted)


def get_false_negatives(true: pd.MultiIndex, predictions: pd.MultiIndex):
    """
    Returns false negative links.

    :param true: Ground truth links
    :param predictions: Predicted links
    """
    return true.difference(predictions)
