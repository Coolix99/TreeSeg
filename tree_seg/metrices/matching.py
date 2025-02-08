# matching.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from tree_seg.metrices.overlap import label_overlap
from tree_seg.metrices.intersection import intersection_over_union, intersection_over_true, intersection_over_pred

# Dictionary of criteria functions
matching_criteria = {
    'iou': intersection_over_union,
    'iot': intersection_over_true,
    'iop': intersection_over_pred,
}

def match_labels(y_true, y_pred, method="1:1", criterion="iou", thresh=0.5):
    """
    Match labels between ground truth and predicted label images based on specified method and criterion.

    Parameters
    ----------
    y_true : ndarray
        Ground truth label image.
    y_pred : ndarray
        Predicted label image.
    method : str
        Matching method to use ('1:1', '1:N', 'N:1').
    criterion : str
        Criterion to use for matching ('iou', 'iot', 'iop').
    thresh : float
        Threshold for the criterion to consider a match valid.
    
    Returns
    -------
    list of tuples
        List of matched label pairs as (true_label, pred_label) tuples.
    """
    # Compute the overlap matrix
    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    # Ignore background
    scores = scores[1:, 1:]  # Remove background from consideration

    if method == "1:1":
        return _match_1_to_1(scores, thresh)
    elif method == "1:N":
        return _match_1_to_N(scores, thresh)
    elif method == "N:1":
        return _match_N_to_1(scores, thresh)
    else:
        raise ValueError(f"Unsupported matching method: {method}")

def _match_1_to_1(scores, thresh):
    """
    Perform 1:1 matching using the given scores and threshold.

    Parameters
    ----------
    scores : ndarray
        Score matrix between true and predicted labels.
    thresh : float
        Minimum score threshold for a valid match.
    
    Returns
    -------
    list of tuples
        List of matched (true_label, pred_label) pairs.
    """
    # Optimal assignment with costs as negative scores
    costs = -(scores >= thresh).astype(float) - scores / (2 * min(scores.shape))
    true_ind, pred_ind = linear_sum_assignment(costs)

    # Filter pairs by threshold
    matches = [(i + 1, j + 1) for i, j in zip(true_ind, pred_ind) if scores[i, j] >= thresh]
    return matches

def _match_1_to_N(scores, thresh):
    """
    Perform 1:N matching where each true label can match multiple predicted labels.

    Parameters
    ----------
    scores : ndarray
        Score matrix between true and predicted labels.
    thresh : float
        Minimum score threshold for a valid match.
    
    Returns
    -------
    list of tuples
        List of matched (true_label, pred_label) pairs.
    """
    matches = []
    for true_label in range(scores.shape[0]):
        matched_preds = np.where(scores[true_label] >= thresh)[0]
        for pred_label in matched_preds:
            matches.append((true_label + 1, pred_label + 1))  # Offset by 1 to account for background
    return matches

def _match_N_to_1(scores, thresh):
    """
    Perform N:1 matching where each predicted label can match multiple true labels.

    Parameters
    ----------
    scores : ndarray
        Score matrix between true and predicted labels.
    thresh : float
        Minimum score threshold for a valid match.
    
    Returns
    -------
    list of tuples
        List of matched (true_label, pred_label) pairs.
    """
    matches = []
    for pred_label in range(scores.shape[1]):
        matched_trues = np.where(scores[:, pred_label] >= thresh)[0]
        for true_label in matched_trues:
            matches.append((true_label + 1, pred_label + 1))  # Offset by 1 to account for background
    return matches
