import torch
import numpy as np
import pytest
from tree_seg.core.neighbor_calculations import calculateBoundaryConnection, shift_tensor

def test_shift_tensor():
    # Create a simple 3D tensor
    mask = torch.tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ], dtype=torch.float32)  # Shape: (2, 2, 2)

    # 1. No shift (should be identical)
    assert torch.equal(shift_tensor(mask, 0, 0, 0), mask)

    # 2. Shift in +z (down)
    expected = torch.tensor([
        [[0, 0], [0, 0]],
        [[1, 2], [3, 4]]
    ], dtype=torch.float32)
    assert torch.equal(shift_tensor(mask, 1, 0, 0), expected)

    # 3. Shift in -z (up)
    expected = torch.tensor([
        [[5, 6], [7, 8]],
        [[0, 0], [0, 0]]
    ], dtype=torch.float32)
    assert torch.equal(shift_tensor(mask, -1, 0, 0), expected)

    # 4. Shift in +y (down)
    expected = torch.tensor([
        [[0, 0], [1, 2]],
        [[0, 0], [5, 6]]
    ], dtype=torch.float32)
    assert torch.equal(shift_tensor(mask, 0, 1, 0), expected)

    # 5. Shift in -y (up)
    expected = torch.tensor([
        [[3, 4], [0, 0]],
        [[7, 8], [0, 0]]
    ], dtype=torch.float32)
    assert torch.equal(shift_tensor(mask, 0, -1, 0), expected)

    # 6. Shift in +x (right)
    expected = torch.tensor([
        [[0, 1], [0, 3]],
        [[0, 5], [0, 7]]
    ], dtype=torch.float32)
    assert torch.equal(shift_tensor(mask, 0, 0, 1), expected)

    # 7. Shift in -x (left)
    expected = torch.tensor([
        [[2, 0], [4, 0]],
        [[6, 0], [8, 0]]
    ], dtype=torch.float32)
    assert torch.equal(shift_tensor(mask, 0, 0, -1), expected)

    # 8. Large shift that clears the tensor
    expected = torch.zeros_like(mask)
    assert torch.equal(shift_tensor(mask, 2, 0, 0), expected)
    assert torch.equal(shift_tensor(mask, 0, 2, 0), expected)
    assert torch.equal(shift_tensor(mask, 0, 0, 2), expected)

def test_single_cube():
    """
    Test a simple 3x3x3 cube where the center voxel is labeled.
    Expected: No connectivity as there are no neighbors.
    """
    mask = torch.zeros((3, 3, 3), dtype=torch.uint8)
    mask[1, 1, 1] = 1  # Single voxel labeled

    connectivity = calculateBoundaryConnection(mask)

    assert connectivity.shape == (6, 3, 3, 3)
    assert torch.sum(connectivity) == 0  # No neighbors should be detected

def test_two_adjacent_voxels():
    """
    Test a 3x3x3 cube where two adjacent voxels are labeled.
    Expected: One connection between the two voxels.
    """
    mask = torch.zeros((3, 3, 3), dtype=torch.uint8)
    mask[1, 1, 1] = 1
    mask[1, 1, 2] = 2  # Adjacent in x-direction

    connectivity = calculateBoundaryConnection(mask)
    # Expect only 2 non-zero entries (one for each voxel)
    assert torch.sum(connectivity) == 2
    assert connectivity[4, 1, 1, 1] == 1  # x+ direction
    assert connectivity[5, 1, 1, 2] == 1  # x- direction

def test_larger_connected_component():
    """
    Test a 3x3x3 cube where a larger block is labeled.
    Expected: Proper connectivity between all touching voxels.
    """
    mask = torch.zeros((3, 3, 3), dtype=torch.uint8)
    mask[1, 1, 1] = 1
    mask[1, 1, 2] = 2
    mask[1, 2, 1] = 3  # Another adjacent voxel in y-direction

    connectivity = calculateBoundaryConnection(mask)

    assert torch.sum(connectivity) == 4  # 3 edges * 2 directions

def test_empty_mask():
    """
    Test an empty mask with no labeled voxels.
    Expected: All connectivity should be zero.
    """
    mask = torch.zeros((3, 3, 3), dtype=torch.uint8)
    connectivity = calculateBoundaryConnection(mask)

    assert torch.sum(connectivity) == 0  # No boundaries should exist

def test_fully_filled_mask():
    """
    Test a fully labeled mask.
    Expected: All internal voxels should be fully connected.
    """
    mask = torch.ones((3, 3, 3), dtype=torch.uint8)
    connectivity = calculateBoundaryConnection(mask)

    # Each internal voxel has 6 neighbors
    assert torch.sum(connectivity) == 0  

@pytest.mark.parametrize("shape", [(5, 5, 5), (2, 2, 2), (4, 4, 4)])
def test_different_shapes(shape):
    """
    Test different shapes to ensure function works on varying sizes.
    """
    mask = torch.ones(shape, dtype=torch.uint8)
    connectivity = calculateBoundaryConnection(mask)

    assert connectivity.shape == (6, *shape)

if __name__ == "__main__":
    pytest.main()
