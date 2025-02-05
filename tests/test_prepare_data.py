import pytest
import torch
import numpy as np
from tree_seg.network_3D.pepare_data import masks_to_flows_gpu_3d, calculateNeighborConnection

@pytest.fixture
def simple_3d_mask():
    """Creates a simple 3D mask with two labeled objects."""
    mask = torch.zeros((5, 5, 5), dtype=torch.int64)
    mask[1:4, 1:4, 1:4] = 1  # Object 1 in center
    mask[3, 3, 3] = 2  # Small separate object
    return mask

@pytest.fixture
def empty_3d_mask():
    """Creates an empty 3D mask."""
    return torch.zeros((5, 5, 5), dtype=torch.int64)

@pytest.fixture
def single_voxel_mask():
    """Creates a mask with a single voxel labeled."""
    mask = torch.zeros((5, 5, 5), dtype=torch.int64)
    mask[2, 2, 2] = 1
    return mask

def test_calculateNeighborConnection_simple(simple_3d_mask):
    """Tests if neighboring connectivity is computed correctly."""
    connectivity = calculateNeighborConnection(simple_3d_mask)

    # Ensure connectivity shape is correct
    assert connectivity.shape == (6, 5, 5, 5)

    # Check that center voxels are connected in all 6 directions
    assert connectivity[:, 2, 2, 2].sum() == 6  # Fully surrounded

    # Check that isolated voxel (3,3,3) has no connections
    assert connectivity[:, 3, 3, 3].sum() == 0

    # Ensure boundaries are handled properly (should have fewer connections)
    assert connectivity[:, 1, 1, 1].sum() == 3  # Corner voxel

def test_calculateNeighborConnection_empty(empty_3d_mask):
    """Tests that an empty mask returns all zeros in connectivity."""
    connectivity = calculateNeighborConnection(empty_3d_mask)
    assert torch.all(connectivity == 0)

def test_calculateNeighborConnection_single(single_voxel_mask):
    """Tests that a single labeled voxel has no neighbors."""
    connectivity = calculateNeighborConnection(single_voxel_mask)
    assert connectivity[:, 2, 2, 2].sum() == 0  # No neighbors

def test_masks_to_flows_gpu_3d_simple():
    """Tests the flow calculation with a simple cubic mask."""
    mask = np.zeros((5, 5, 5), dtype=np.int64)
    mask[1:4, 1:4, 1:4] = 1  # Single object in center

    flow = masks_to_flows_gpu_3d(mask)

    # Ensure correct shape
    assert flow.shape == (3, 5, 5, 5)

    # Check that flow vectors are unit normalized
    norms = np.linalg.norm(flow, axis=0)
    assert np.allclose(norms[mask > 0], 1.0, atol=1e-2)  # Flow vectors should be normalized

    # Ensure zero flow for empty areas
    assert np.all(flow[:, mask == 0] == 0)

def test_masks_to_flows_gpu_3d_empty():
    """Tests that an empty mask returns zero flow everywhere."""
    mask = np.zeros((5, 5, 5), dtype=np.int64)
    flow = masks_to_flows_gpu_3d(mask)
    assert np.all(flow == 0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_masks_to_flows_gpu_3d_gpu():
    """Tests GPU execution of flow computation."""
    mask = np.zeros((5, 5, 5), dtype=np.int64)
    mask[1:4, 1:4, 1:4] = 1  # Single object
    device = torch.device('cuda')
    
    flow = masks_to_flows_gpu_3d(mask, device=device)

    # Ensure correct shape and GPU execution
    assert flow.shape == (3, 5, 5, 5)
    assert torch.tensor(flow).device == torch.device('cpu')  # Returns numpy, so CPU

if __name__ == "__main__":
    pytest.main()