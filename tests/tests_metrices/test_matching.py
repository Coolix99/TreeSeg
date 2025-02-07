import numpy as np
import pytest
from tree_seg.metrices.matching import match_labels

def test_1_to_1_matching():
    y_true = np.array([
        [1, 1, 0],
        [2, 0, 0],
        [2, 2, 0]
    ])
    
    y_pred = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [2, 2, 0]
    ])
    
    # Test 1:1 matching with IoU criterion
    expected_matches = [(1, 1), (2, 2)]
    result = match_labels(y_true, y_pred, method="1:1", criterion="iou", thresh=0.6)
    assert result == expected_matches, f"1:1 matching failed. Expected {expected_matches}, got {result}"

def test_1_to_N_matching():
    y_true = np.array([
        [1, 1, 0],
        [2, 2, 0],
        [2, 2, 0]
    ])
    
    y_pred = np.array([
        [1, 3, 0],
        [0, 2, 2],
        [0, 2, 2]
    ])
    
    # Test 1:N matching with IoT criterion
    expected_matches = [(1, 1),(1,3)]
    result = match_labels(y_true, y_pred, method="1:N", criterion="iop", thresh=0.6)
    assert result == expected_matches, f"1:N matching failed. Expected {expected_matches}, got {result}"

def test_N_to_1_matching():
    y_true = np.array([
        [3, 1, 0],
        [0, 2, 0],
        [0, 2, 2]
    ])
    
    y_pred = np.array([
        [1, 1, 0],
        [2, 2, 2],
        [2, 0, 0]
    ])
    
    # Test N:1 matching with IoP criterion
    expected_matches = [(1, 1),(3,1)]
    result = match_labels(y_true, y_pred, method="N:1", criterion="iot", thresh=0.6)
    assert result == expected_matches, f"N:1 matching failed. Expected {expected_matches}, got {result}"

if __name__ == "__main__":
    pytest.main()
