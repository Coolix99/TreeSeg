import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tree_seg.network_3D.UNet3D import UNet3D  # Import the trained model

class ApplyDataset(Dataset):
    """Dataset for applying the UNet3D model on a 3D image."""
    def __init__(self, image, profile, patch_size, overlap):
        self.image = image
        self.profile = profile
        self.patch_size = patch_size
        self.overlap = overlap
        self.patches, self.positions = self._extract_patches()

    def _extract_patches(self):
        """Extract patches from the 3D image with overlapping regions."""
        patches = []
        positions = []
        stride = tuple(s - o for s, o in zip(self.patch_size, self.overlap))
        
        # Ensure full coverage by extending to cover the last region
        for z in range(0, self.image.shape[0], stride[0]):
            for y in range(0, self.image.shape[1], stride[1]):
                for x in range(0, self.image.shape[2], stride[2]):
                    # Ensure patch fits in the image
                    z_start = min(z, self.image.shape[0] - self.patch_size[0])
                    y_start = min(y, self.image.shape[1] - self.patch_size[1])
                    x_start = min(x, self.image.shape[2] - self.patch_size[2])

                    patch = self.image[z_start:z_start + self.patch_size[0], 
                                    y_start:y_start + self.patch_size[1], 
                                    x_start:x_start + self.patch_size[2]]
                    
                    if np.any(patch):  # Only process patches with non-zero content
                        patches.append(patch)
                        positions.append(np.array((z_start, y_start, x_start)))

        return patches, positions

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        position = self.positions[idx]
        patch = np.expand_dims(patch, axis=0)  # Add channel dimension
        return torch.from_numpy(patch).float(), torch.from_numpy(self.profile).float(), position

def apply_model(config, image, profile):
    """Apply the trained UNet3D model to an input 3D image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = UNet3D(n_channels=1, context_size=config["context_size"], patch_size=config["patch_size"]).to(device)

    # Correct way to load weights
    checkpoint = torch.load(config["model_path"], map_location=device, weights_only=True)  
    model.load_state_dict(checkpoint["model_state_dict"])  # Correct loading

    model.eval()


    # Prepare dataset and dataloader
    overlap = (config["patch_size"] // 2,) * 3  # 50% overlap
    dataset = ApplyDataset(image, profile, (config["patch_size"],) * 3, overlap)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["n_cores"])

    processed_patches = []
    positions = []

    # Process each patch
    with torch.no_grad():
        for patches, profiles, pos in dataloader:
            patches, profiles = patches.to(device), profiles.to(device)
            seg_logits, pred_flows, neighbor_logits = model(patches, profiles)
          
            pred_flows = pred_flows.cpu().numpy()
            seg_logits = seg_logits.cpu().numpy()[:,0,:,:,:]
            neighbor_logits = neighbor_logits.cpu().numpy()
           
            for i in range(len(patches)):
                processed_patches.append((seg_logits[i], pred_flows[i], neighbor_logits[i]))
                positions.append(pos[i].numpy())

    # Reconstruct full image
    pred_mask, pred_flows, neighbor_preds = reconstruct_image(image.shape, config["patch_size"], processed_patches, positions)

    return pred_mask, pred_flows, neighbor_preds

def reconstruct_image(image_shape, patch_size, patches, positions):
    """Reconstruct the full 3D image from overlapping patches."""
    reconstructed_seg = np.zeros(image_shape, dtype=np.float32)
    reconstructed_flow = np.zeros((3,) + image_shape, dtype=np.float32)
    reconstructed_neighbors = np.zeros((6,) + image_shape, dtype=np.float32)
    counts = np.zeros(image_shape, dtype=np.int64)

    for (seg_patch, flow_patch, neighbor_patch), pos in zip(patches, positions):
        z, y, x = pos
        z_slice, y_slice, x_slice = slice(z, z + patch_size), slice(y, y + patch_size), slice(x, x + patch_size)

        reconstructed_seg[z_slice, y_slice, x_slice] += seg_patch
        reconstructed_flow[:, z_slice, y_slice, x_slice] += flow_patch
        reconstructed_neighbors[:, z_slice, y_slice, x_slice] += neighbor_patch
        counts[z_slice, y_slice, x_slice] += 1

    # Normalize overlapping areas
    nonzero_mask = counts > 0
    reconstructed_seg[nonzero_mask] /= counts[nonzero_mask]
    reconstructed_flow[:, nonzero_mask] /= counts[nonzero_mask]
    reconstructed_neighbors[:, nonzero_mask] /= counts[nonzero_mask]

    # Normalize flow vectors
    flow_magnitudes = np.linalg.norm(reconstructed_flow, axis=0)
    nonzero_flows_mask = flow_magnitudes > 0
    reconstructed_flow[:, nonzero_flows_mask] /= flow_magnitudes[nonzero_flows_mask]

    # Convert segmentation to binary
    reconstructed_seg = (reconstructed_seg > 0.0).astype(bool)
    return reconstructed_seg, reconstructed_flow, reconstructed_neighbors
