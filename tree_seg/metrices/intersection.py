import numpy as np

"""
This module provides intersection-based metrics for evaluating the overlap between labeled segmentation images.
Metrics include:
- Intersection over Union (IoU): Measures the ratio of overlap to the union of true and predicted labels.
- Intersection over True (IoT): Measures the ratio of overlap to the area of the true label.
- Intersection over Prediction (IoP): Measures the ratio of overlap to the area of the predicted label.
"""

def intersection_over_union(overlap):
    """
    Compute Intersection over Union (IoU) based on the overlap matrix.

    Parameters
    ----------
    overlap : ndarray
        Overlap matrix where each entry (i, j) represents the count of pixels where label `i` in `x`
        overlaps with label `j` in `y`.
    
    Returns
    -------
    ndarray
        IoU matrix, where each entry (i, j) is the IoU score for the label pair (i, j).
    """
    if np.sum(overlap) == 0:
        return overlap

    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)  # Sum over rows
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)  # Sum over columns
    denominator = n_pixels_pred + n_pixels_true - overlap

    # Avoid division by zero by only dividing where the denominator is not zero
    iou_matrix = np.divide(overlap, denominator, where=(denominator != 0))
    return iou_matrix

def intersection_over_true(overlap):
    """
    Compute Intersection over True (IoT) based on the overlap matrix.
    
    Parameters
    ----------
    overlap : ndarray
        Overlap matrix where each entry (i, j) represents the count of pixels where label `i` in `x`
        overlaps with label `j` in `y`.
    
    Returns
    -------
    ndarray
        IoT matrix, where each entry (i, j) is the IoT score for the label pair (i, j).
    """
    if np.sum(overlap) == 0:
        return overlap

    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)  # Sum over columns (true labels)
    iot_matrix = np.divide(overlap, n_pixels_true, where=(n_pixels_true != 0))
    return iot_matrix

def intersection_over_pred(overlap):
    """
    Compute Intersection over Prediction (IoP) based on the overlap matrix.
    
    Parameters
    ----------
    overlap : ndarray
        Overlap matrix where each entry (i, j) represents the count of pixels where label `i` in `x`
        overlaps with label `j` in `y`.
    
    Returns
    -------
    ndarray
        IoP matrix, where each entry (i, j) is the IoP score for the label pair (i, j).
    """
    if np.sum(overlap) == 0:
        return overlap

    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)  # Sum over rows (predicted labels)
    iop_matrix = np.divide(overlap, n_pixels_pred, where=(n_pixels_pred != 0))
    return iop_matrix