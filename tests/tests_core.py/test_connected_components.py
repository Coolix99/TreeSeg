import numpy as np
from numpy.testing import assert_array_equal
import pytest
from tree_seg.core.pre_segmentation import connected_components_3D, euler_connected_components_3D

def relabel_sequential(labels):
    """
    Relabels connected component labels sequentially starting from 1,
    in the order they first appear in the flattened array.
    
    Parameters:
        labels (np.ndarray): A labeled image array with non-sequential labels.

    Returns:
        np.ndarray: A relabeled array with labels starting from 1, in the order of appearance.
    """
    # Flatten the labels and get unique labels in the order they appear, excluding 0
    flat_labels = labels.flatten()
    unique_labels = [label for label in flat_labels if label != 0]
    
    # Maintain the order of first appearance in the flattened array
    seen = set()
    ordered_labels = []
    for label in unique_labels:
        if label not in seen:
            ordered_labels.append(label)
            seen.add(label)

    # Create a mapping from old labels to new sequential labels based on order of appearance
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(ordered_labels, start=1)}
    
    # Apply the mapping to create a new array with sequential labels
    relabeled = np.zeros_like(labels, dtype=np.int32)
    for old_label, new_label in label_mapping.items():
        relabeled[labels == old_label] = new_label
    
    return relabeled

def test_connected_components_3D_x_direction_component():
    """
    Test `connected_components_3D` with components connected only along the X direction.
    Each YZ slice should be independent.
    """
    N = 5
    mask = np.ones((N, N, N), dtype=np.int32)
    vector_field = np.zeros((3, N, N, N), dtype=np.float32)
    
    # Only connect in the X-direction
    vector_field[0, :, :, :] = -1  # X direction
    vector_field[1, :, :, :] = 0   # Y direction
    vector_field[2, :, :, :] = 0   # Z direction

    # Expected labels: Each YZ slice should be independently labeled
    expected_labels = np.zeros((N, N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            expected_labels[:, i, j] = i*N+j+1

    labels = connected_components_3D(mask, vector_field)
    labels = relabel_sequential(labels)
    print('--')
    print(labels)
    assert_array_equal(labels, relabel_sequential(expected_labels))

def test_connected_components_3D_disconnected():
    """Test `connected_components_3D` with multiple disconnected components """
    mask = np.array([
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ], dtype=np.int32)

    vector_field = np.zeros((3, 3, 3, 3), dtype=np.float32)

    expected_labels = np.array([
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 2, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 3]]
    ], dtype=np.int32)

    labels = connected_components_3D(mask, vector_field)
    labels = relabel_sequential(labels)
    assert_array_equal(labels, relabel_sequential(expected_labels))

def test_connected_components_3D_different_connectivity():
    """
    Test `connected_components_3D` 
    """
    mask = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[1, 0, 0],
         [0, 1, 1],
         [0, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ], dtype=np.int32)

    vector_field = np.ones((3, 3, 3, 3), dtype=np.float32)

    # Expected result with 6-connectivity
    expected_labels_6 = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[2, 0, 0],
         [0, 1, 3],
         [0, 4, 5]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 5]]
    ], dtype=np.int32)


  
    # Test 6-connectivity
    labels_6 = connected_components_3D(mask, vector_field)
    labels_6 = relabel_sequential(labels_6)
    assert_array_equal(labels_6, relabel_sequential(expected_labels_6))

def test_euler_connected_components_3D_x_direction_component():
    """
    Test `euler_connected_components_3D` with components connected only along the X direction.
    Each YZ slice should be independent.
    """
    N = 5
    mask = np.ones((N, N, N), dtype=np.int32)
    vector_field = np.zeros((3, N, N, N), dtype=np.float32)
    
    # Flow is only in the X-direction
    vector_field[0, :, :, :] = 1  # Move right
    vector_field[1, :, :, :] = 0  # No movement in Y
    vector_field[2, :, :, :] = 0  # No movement in Z

    expected_labels = np.zeros((N, N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            expected_labels[:, i, j] = i * N + j + 1

    labels = euler_connected_components_3D(mask, vector_field, step_size=0.3, N_steps=30)

    labels = relabel_sequential(labels)
    assert_array_equal(labels, relabel_sequential(expected_labels))

def test_euler_connected_components_3D_disconnected():
    """
    Test `euler_connected_components_3D` with multiple disconnected components.
    """
    mask = np.array([
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ], dtype=np.int32)

    vector_field = np.zeros((3, 3, 3, 3), dtype=np.float32)  # No flow → No connectivity

    expected_labels = np.array([
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 2, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 3]]
    ], dtype=np.int32)

    labels = euler_connected_components_3D(mask, vector_field, step_size=0.3, N_steps=30)
    labels = relabel_sequential(labels)
    assert_array_equal(labels, relabel_sequential(expected_labels))

def test_euler_connected_components_3D_different_connectivity():
    """
    Test `euler_connected_components_3D` where the flow moves diagonally and components merge differently.
    """
    mask = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[1, 0, 0],
         [0, 1, 1],
         [0, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ], dtype=np.int32)

    vector_field = np.ones((3, 3, 3, 3), dtype=np.float32)  # All directions move positively
    vector_field[:,1,1,1]=0
    # Expected result should differ from standard connectivity due to flow influence
    expected_labels_euler = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],

        [[1, 0, 0],
         [0, 2, 1],
         [0, 1, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]]
    ], dtype=np.int32)

    labels_euler = euler_connected_components_3D(mask, vector_field, step_size=0.3, N_steps=30)
    labels_euler = relabel_sequential(labels_euler)
    assert_array_equal(labels_euler, relabel_sequential(expected_labels_euler))

if __name__ == "__main__":
    pytest.main([__file__])


