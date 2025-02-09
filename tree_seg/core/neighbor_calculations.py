import torch
import torch.nn.functional as F

def shift_tensor(mask, dz, dy, dx):
    """
    Shifts a 3D tensor along z, y, and x axes while padding with zeros.

    Args:
        mask (torch.Tensor): 3D tensor of shape (D, H, W).
        dz (int): Shift along depth (z-axis).
        dy (int): Shift along height (y-axis).
        dx (int): Shift along width (x-axis).

    Returns:
        torch.Tensor: Shifted tensor with zero padding.
    """
    D, H, W = mask.shape

    # Padding before shifting
    pad_z = (max(dz, 0), max(-dz, 0))
    pad_y = (max(dy, 0), max(-dy, 0))
    pad_x = (max(dx, 0), max(-dx, 0))

    # Apply zero-padding
    padded = F.pad(mask, (pad_x[0], pad_x[1], pad_y[0], pad_y[1], pad_z[0], pad_z[1]), value=0)

    # Extract valid region
    new_mask = padded[
        pad_z[1] : pad_z[1] + D,
        pad_y[1] : pad_y[1] + H,
        pad_x[1] : pad_x[1] + W
    ]

    return new_mask


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
        shifted_mask = shift_tensor(mask, dz, dy, dx)
        
        # Check if the voxel has the same label as its shifted neighbor
        other_label = (mask != shifted_mask) & (mask > 0) & (shifted_mask > 0) # Ignore background (0)
    
        
        # Store result in the corresponding channel
        connectivity[i] = other_label

    return connectivity
