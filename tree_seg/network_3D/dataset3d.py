import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import rotate

def random_rotation_and_mirror(image, mask, flow, neighbors):
    """Apply random 90-degree rotation and mirroring to data for augmentation."""
    k = np.random.randint(0, 4)
    
    # Rotate images, masks, and flow fields
    image = rotate(image, angle=90*k, axes=(2, 3), reshape=False)
    mask = rotate(mask, angle=90*k, axes=(1, 2), reshape=False)
    flow = rotate(flow, angle=90*k, axes=(2, 3), reshape=False)
    neighbors = rotate(neighbors, angle=90*k, axes=(2, 3), reshape=False)

    # Compute 2D rotation matrix
    cos_angle = np.cos(np.deg2rad(90*k))
    sin_angle = np.sin(np.deg2rad(90*k))
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # Rotate flow vectors
    flow_vectors = flow[1:3].reshape(2, -1)
    rotated_vectors = np.dot(rotation_matrix, flow_vectors).reshape(2, *flow.shape[1:])
    flow[1:3] = rotated_vectors

    # Swap neighbor connections based on rotation
    if k == 1:  # 90 degrees
        neighbors = neighbors[[0, 1, 4, 5, 3, 2]] 
    elif k == 2:  # 180 degrees
        neighbors = neighbors[[0, 1, 3, 2, 5, 4]]  
    elif k == 3:  # 270 degrees
        neighbors = neighbors[[0, 1, 5, 4, 2, 3]]  
    
    # Apply random flipping
    if np.random.rand() > 0.5:
        image, mask, flow, neighbors = np.flip(image, axis=2), np.flip(mask, axis=1), np.flip(flow, axis=2), np.flip(neighbors, axis=2)
        flow[1, :, :, :] = -flow[1, :, :, :]
        neighbors = neighbors[[0, 1, 3, 2, 4, 5]]

    if np.random.rand() > 0.5:
        image, mask, flow, neighbors= np.flip(image, axis=1), np.flip(mask, axis=0), np.flip(flow, axis=1), np.flip(neighbors, axis=1)
        flow[0, :, :, :] = -flow[0, :, :, :]
        neighbors = neighbors[[1, 0, 2, 3, 4, 5]]

    return image, mask, flow, neighbors

class Dataset3D(Dataset):
    def __init__(self, images, masks, flows, contexts,neighbors, patch_size=(64, 64, 64), min_nonzero=0.05, transform=random_rotation_and_mirror):
        """
        3D Dataset for volumetric images, segmentation masks, and flow fields.

        Args:
            images (list of 3D np.array): Input volume images.
            masks (list of 3D np.array): Segmentation masks.
            flows (list of 3D np.array): Flow fields (3D vector fields).
            contexts (list of 1D np.array or None): Contextual vectors per sample. If None, replaces with 0.5 values.
            patch_size (tuple): Patch size (Depth, Height, Width).
            min_nonzero (float): Minimum fraction of nonzero pixels in a patch to be valid.
            transform (callable): Data augmentation function.
        """
        self.images = images
        self.masks = masks
        self.flows = flows
        self.contexts = contexts
        self.neighbors = neighbors
        self.patch_size = patch_size
        self.min_nonzero = min_nonzero
        self.transform = transform


        # Extract valid patches
        self.patches = self._extract_patches()

    def _extract_patches(self):
        """Extracts patches from the dataset ensuring they meet nonzero pixel criteria."""
        patches = []
        for img, mask, flow, context,neighbors in zip(self.images, self.masks, self.flows, self.contexts, self.neighbors):
            pz, py, px = self.patch_size
            nz, ny, nx = img.shape
            for z in range(0, nz - pz + 1, pz):
                for y in range(0, ny - py + 1, py):
                    for x in range(0, nx - px + 1, px):
                        img_patch = img[z:z + pz, y:y + py, x:x + px]

                        # Skip patches with too few nonzero pixels
                        if np.sum(img_patch > 0) < int(self.min_nonzero * np.prod(self.patch_size)):
                            continue

                        mask_patch = mask[z:z + pz, y:y + py, x:x + px]
                        flow_patch = flow[:, z:z + pz, y:y + py, x:x + px]
                        neighbors_patch = neighbors[:, z:z + pz, y:y + py, x:x + px]

                        patches.append((img_patch, mask_patch, flow_patch, context,neighbors_patch))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """Retrieves and processes a single sample."""
        img_patch, mask_patch, flow_patch, context, neighbors_patch  = self.patches[idx]
        
        # Expand image dimensions (add channel axis)
        img_patch = np.expand_dims(img_patch, axis=0)

        # Apply transformations (data augmentation)
        if self.transform:
            img_patch, mask_patch, flow_patch, neighbors_patch = self.transform(img_patch, mask_patch, flow_patch,neighbors_patch)

        # Convert to PyTorch tensors
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch)).float()
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch)).bool()
        flow_patch = torch.from_numpy(np.ascontiguousarray(flow_patch)).float()
        neighbors_patch = torch.from_numpy(np.ascontiguousarray(neighbors_patch)).bool()
        context = torch.from_numpy(np.ascontiguousarray(context)).float()

        return img_patch, mask_patch, flow_patch, context, neighbors_patch
