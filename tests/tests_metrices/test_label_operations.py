import numpy as np
from  tree_seg.metrices.label_operations import relabel_sequentially_3D,relabel_sequentially

def test_relabel_sequentially():
    # Create a sample 2D array with non-sequential labels
    labels_2d = np.array([
        [1, 2, 2, 0],
        [3, 0, 3, 1],
        [0, 4, 4, 3],
        [4, 2, 1, 0]
    ])
    
    # Expected output after relabeling: sequential labels starting from 1
    expected_2d = np.array([
        [1, 2, 2, 0],
        [3, 0, 3, 1],
        [0, 4, 4, 3],
        [4, 2, 1, 0]
    ])
    
    # Call the relabel_sequentially function
    relabeled_2d = relabel_sequentially(labels_2d)
    
    # Check if the output matches the expected output
    assert np.array_equal(relabeled_2d, expected_2d), "Test failed for 2D relabeling"
    print("2D test passed.")

def test_relabel_sequentially_3D():
    # Create a sample 3D array with non-sequential labels
    labels_3d = np.array([
        [
            [1, 2, 0],
            [3, 0, 4]
        ],
        [
            [4, 0, 3],
            [2, 1, 4]
        ]
    ])
    
    # Expected output after relabeling: sequential labels starting from 1
    expected_3d = np.array([
        [
            [1, 2, 0],
            [3, 0, 4]
        ],
        [
            [4, 0, 3],
            [2, 1, 4]
        ]
    ])
    
    # Call the relabel_sequentially_3D function
    relabeled_3d = relabel_sequentially_3D(labels_3d)
    
    # Check if the output matches the expected output
    assert np.array_equal(relabeled_3d, expected_3d), "Test failed for 3D relabeling"
    print("3D test passed.")

# Run tests
if __name__ == "__main__":
    #test_relabel_sequentially()
    test_relabel_sequentially_3D()
