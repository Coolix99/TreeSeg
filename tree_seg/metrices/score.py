import numpy as np
from tree_seg.metrices.utils import _safe_divide
from tree_seg.metrices.matching import match_labels

def calculate_scores(y_true, y_pred, thresholds=[0.5], criterion='iou', method='1:1'):
    """
    Calculate precision, recall, F1, and panoptic quality for different thresholds.

    Parameters
    ----------
    y_true : ndarray
        Ground truth label image.
    y_pred : ndarray
        Predicted label image.
    thresholds : list of floats
        List of thresholds to calculate scores for.
    criterion : str
        Criterion to use for matching ('iou', 'iot', 'iop').
    method : str
        Matching method to use ('1:1', '1:N', 'N:1').

    Returns
    -------
    dict
        Dictionary of scores for each threshold, including precision, recall, F1, and panoptic quality.
    """
    scores_by_threshold = {}

    for thresh in thresholds:
        # Get matches using the specified matching method and criterion
        matches = match_labels(y_true, y_pred, method=method, criterion=criterion, thresh=thresh)

        # Calculate true positives, false positives, and false negatives
        tp = len(matches)  # True Positives: the matched pairs
        n_true_labels = len(np.unique(y_true)) - 1  # Exclude background (0 label)
        n_pred_labels = len(np.unique(y_pred)) - 1  # Exclude background (0 label)
        fp = n_pred_labels - tp  # False Positives
        fn = n_true_labels - tp  # False Negatives

        # Calculate precision, recall, F1 score, and panoptic quality
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1_score = _safe_divide(2 * tp, 2 * tp + fp + fn)
        panoptic_quality = _safe_divide(tp, tp + fp / 2 + fn / 2)

        # Store the scores for the current threshold
        scores_by_threshold[thresh] = {
            'precision': precision,
            'recall': recall,
            'f1': f1_score,
            'panoptic_quality': panoptic_quality,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return scores_by_threshold
