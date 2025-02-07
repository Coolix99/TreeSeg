import numpy as np
import pytest
from tree_seg.metrices.overlap import label_overlap

def test_label_overlap():
    x = np.array([
        [1, 1, 0],
        [2, 0, 0],
        [2, 2, 0]
    ])
    
    y = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [2, 2, 2]
    ])
    
    overlap_matrix = label_overlap(x, y)

    expected = np.array([
        [3, 0, 1],
        [1, 1, 0],
        [0, 1, 2]
    ], dtype=np.uint32)
    
    assert np.array_equal(overlap_matrix, expected)
    
    # Test shape mismatch error
    with pytest.raises(ValueError):
        label_overlap(x, np.ones((2, 2), dtype=int))

    # Test with sequential and non-sequential labels
    non_sequential_y = np.array([
        [1, 0, 0],
        [1, 4, 0],
        [2, 2, 2]
    ])
    with pytest.raises(ValueError):
        label_overlap(x, non_sequential_y, check=True)

if __name__ == "__main__":
    print("Running test_label_overlap...")
    test_label_overlap()
    print("All tests passed!")