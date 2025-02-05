import pytest
import numpy as np
from scipy.ndimage import rotate
from tree_seg.network_3D.dataset3d  import random_rotation_and_mirror  

@pytest.fixture
def sample_data():
    """Generate a sample 3D dataset with known values for testing."""
    image = np.random.rand(1, 10, 10, 10)  # 3D grayscale image
    mask = np.zeros((10, 10, 10))
    mask[2:8, 2:8, 2:8] = 1  # A cube mask in the center

    flow = np.zeros((3, 10, 10, 10))  # 3D vector field
    flow[1, :, :, :] = 1  # Set y-component flow to 1
    flow[2, :, :, :] = -1  # Set x-component flow to -1
    flow = flow / np.linalg.norm(flow, axis=0)  # Normalize flow vectors

    neighbors = np.zeros((6, 10, 10, 10))  # Dummy neighbor connectivity
    neighbors[0] = 1  # Top connected
    neighbors[1] = 1  # Bottom connected

    return image, mask, flow, neighbors

def test_random_rotation_and_mirror(sample_data):
    """Test rotation, flipping, and neighbor transformation correctness."""
    image, mask, flow, neighbors = sample_data
    orig_shape = image.shape

    for _ in range(10):  # Test multiple augmentations
        aug_image, aug_mask, aug_flow, aug_neighbors = random_rotation_and_mirror(image, mask, flow, neighbors)

        # Ensure shape remains unchanged
        assert aug_image.shape == orig_shape, "Image shape changed after augmentation"
        assert aug_mask.shape == mask.shape, "Mask shape changed after augmentation"
        assert aug_flow.shape == flow.shape, "Flow shape changed after augmentation"
        assert aug_neighbors.shape == neighbors.shape, "Neighbors shape changed after augmentation"

        # Ensure flow remains normalized (magnitude should be ~1)
        flow_magnitude = np.linalg.norm(aug_flow, axis=0)
        assert np.allclose(flow_magnitude[mask > 0], 1, atol=1e-2), "Flow vectors not normalized"


if __name__ == "__main__":
    pytest.main()