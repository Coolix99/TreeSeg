import torch

def calculateBoundaryConnection(mask):
    """
    Calculates neighbor connectivity in a 3D mask.
    
    Args:
        mask (torch.Tensor): 3D tensor of shape (D, H, W) representing labeled masks.
        
    Returns:
        torch.Tensor: 6-channel binary tensor (6, D, H, W) indicating connectivity.
    """
    device = mask.device
    D, H, W = mask.shape

    # Initialize empty tensor for neighbor connectivity (6 directions)
    connectivity = torch.zeros((6, D, H, W), dtype=torch.uint8, device=device)

    # Define shifts in 6 directions (z, y, x) -> (depth, height, width)
    shifts = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    # Iterate over shifts
    for i, (dz, dy, dx) in enumerate(shifts):
        shifted_mask = torch.roll(mask, shifts=(dz, dy, dx), dims=(0, 1, 2))
        
        # Check if the voxel has the same label as its shifted neighbor
        other_label = (mask != shifted_mask) & (mask > 0) & (shifted_mask > 0) # Ignore background (0)
    
        
        # Store result in the corresponding channel
        connectivity[i] = other_label

    return connectivity
