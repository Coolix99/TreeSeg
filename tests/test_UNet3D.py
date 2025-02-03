import torch
import pytest
from tree_seg.network_3D.UNet3D import UNet3D  # Assuming the UNet3D class is saved in unet3d.py

@pytest.mark.parametrize("batch_size", [1, 2, 4])  # Test different batch sizes
@pytest.mark.parametrize("patch_size", [32, 64])  # Test different valid patch sizes
def test_unet3d(batch_size, patch_size):
    n_channels = 1  # Example: single-channel input (e.g., grayscale)
    context_size = 100  # Example context vector size

    # Ensure patch_size is valid (should be a power of 2 and >= 8)
    assert patch_size >= 8 and (patch_size & (patch_size - 1)) == 0, "Invalid patch size"

    model = UNet3D(n_channels=n_channels, context_size=context_size, patch_size=patch_size)

    # Create random input tensor (Batch, Channels, Depth, Height, Width)
    x = torch.randn(batch_size, n_channels, patch_size, patch_size, patch_size)
    context_vector = torch.randn(batch_size, context_size)

    # Forward pass
    seg_logits, flow_field, neighbor_logits = model(x, context_vector)

    # Check output dimensions
    assert seg_logits.shape == (batch_size, 1, patch_size, patch_size, patch_size), "Segmentation output shape mismatch"
    assert flow_field.shape == (batch_size, 3, patch_size, patch_size, patch_size), "Flow field output shape mismatch"
    assert neighbor_logits.shape == (batch_size, 6, patch_size, patch_size, patch_size), "Neighbor output shape mismatch"

    # Ensure flow is normalized
    flow_magnitude = torch.norm(flow_field, dim=1)
    assert torch.allclose(flow_magnitude, torch.ones_like(flow_magnitude), atol=1e-3), "Flow field is not normalized"

if __name__ == "__main__":
    pytest.main()
