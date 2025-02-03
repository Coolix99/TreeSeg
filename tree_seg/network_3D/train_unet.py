import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from nuclei_dataset import NucleiDataset  # Ensure this file exists
from tree_seg.network_3D.UNet3D import UNet3D  # Ensure this file exists

# ---------------------- Configuration ---------------------- #
CONFIG = {
    "batch_size": 2,
    "num_epochs": 150,
    "learning_rate": 0.0005,
    "context_size": 8,
    "patch_size": 64,  # Must be power of 2
    "n_cores": 4,
    "checkpoint_dir": "checkpoints",
    "dataset_paths": {
        "nuclei": "data/nuclei.npy",
        "masks": "data/masks.npy",
        "flows": "data/flows.npy",
        "profiles": "data/profiles.npy"
    }
}

# ---------------------- Loss Functions ---------------------- #
def angle_loss(pred_flow, true_flow, mask):
    """Angle loss between predicted and true flow field."""
    cosine_similarity = torch.sum(pred_flow * true_flow, dim=1)
    angle_loss = 1 - cosine_similarity
    angle_loss = angle_loss * mask
    return angle_loss.sum() / mask.sum()

def masked_cross_entropy_loss(logits, target, mask, criterion):
    """Masked cross-entropy loss for segmentation."""
    mask = mask.float()
    loss = criterion(logits, target.long())
    return (loss * mask).sum() / mask.sum()

# ---------------------- Training Function ---------------------- #
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = NucleiDataset(
        np.load(CONFIG["dataset_paths"]["nuclei"]),
        np.load(CONFIG["dataset_paths"]["masks"]),
        np.load(CONFIG["dataset_paths"]["flows"]),
        np.load(CONFIG["dataset_paths"]["profiles"]),
        patch_size=(CONFIG["patch_size"], CONFIG["patch_size"], CONFIG["patch_size"])
    )

    # Train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["n_cores"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["n_cores"], pin_memory=True)

    # Initialize model
    model = UNet3D(n_channels=1, context_size=CONFIG["context_size"], patch_size=CONFIG["patch_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion_segmentation = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0], dtype=torch.float32).to(device))

    # Checkpoint handling
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "unet3d_best.pth")

    # Training loop
    best_val_loss = np.inf
    start_time = time.time()

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_segmentation, device)
        val_loss = validate_one_epoch(model, val_loader, criterion_segmentation, device)

        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            elapsed_time = time.time() - start_time
            save_checkpoint(model, checkpoint_path, epoch, train_loss, val_loss, elapsed_time)

# ---------------------- Training Step ---------------------- #
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks, flows, prof in train_loader:
        images, masks, flows, prof = images.to(device), masks.to(device), flows.to(device), prof.to(device)
        optimizer.zero_grad()

        seg_logits, pred_flows, neighbor_logits = model(images, prof)

        mask = images > 0  # Mask based on intensities
        loss_segmentation = masked_cross_entropy_loss(seg_logits, masks, mask, criterion)
        loss_flow = angle_loss(pred_flows, flows, mask)
        loss = loss_segmentation + loss_flow

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

# ---------------------- Validation Step ---------------------- #
def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks, flows, prof in val_loader:
            images, masks, flows, prof = images.to(device), masks.to(device), flows.to(device), prof.to(device)

            seg_logits, pred_flows, neighbor_logits = model(images, prof)

            mask = images > 0
            loss_segmentation = masked_cross_entropy_loss(seg_logits, masks, mask, criterion)
            loss_flow = angle_loss(pred_flows, flows, mask)
            loss = loss_segmentation + loss_flow

            val_loss += loss.item()

    return val_loss / len(val_loader)

# ---------------------- Save Model ---------------------- #
def save_checkpoint(model, path, epoch, train_loss, val_loss, elapsed_time):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'elapsed_time': elapsed_time
    }
    torch.save(checkpoint, path)
    print(f"✅ Model saved at {path} (Epoch {epoch}, Loss: {val_loss:.4f})")

# ---------------------- Run Training ---------------------- #
if __name__ == "__main__":
    train_model()
