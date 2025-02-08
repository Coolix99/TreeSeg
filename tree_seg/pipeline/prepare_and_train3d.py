import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle

from tree_seg.network_3D.pepare_data import calculateFlow, calculateNeighborConnection  
from tree_seg.network_3D.train_unet import train_model  
from tree_seg.core.pre_segmentation import connected_components_3D
from tree_seg.pipeline.apply_model3d import main as apply_model
from tree_seg.pipeline.visualize import visualize_results
from tree_seg.metrices.label_operations import relabel_sequentially_3D
from tree_seg.metrices.matching import match_labels

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

 
### 🔹 COMMON UTILITY FUNCTIONS

def check_required_files(file_paths, subfolder):
    """Checks if all required files exist before processing."""
    missing_files = [p for p in file_paths if not os.path.exists(p)]
    if missing_files:
        logging.warning(f"Skipping {subfolder}, missing files: {missing_files}")
        return False
    return True

def fit_and_save_kde(data, filename, results_folder):
    """Fits a KDE model and saves it as a pickle file."""
    if len(data) == 0:
        logging.warning(f"Skipping KDE fitting for {filename}, empty dataset.")
        return None

    kde = gaussian_kde(data)
    kde_path = os.path.join(results_folder, filename)
    
    with open(kde_path, 'wb') as f:
        pickle.dump(kde, f)
    
    logging.info(f"📁 Saved KDE model: {kde_path}")
    return kde

def plot_distribution(data, kde, title, color, save_path):
    """Plots histogram and KDE curve, saves the plot."""
    if len(data) == 0 or kde is None:
        logging.warning(f"Skipping {title}, missing data or KDE.")
        return
    
    x_vals = np.linspace(min(data), max(data), 200)
    kde_vals = kde(x_vals)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.5, color=color, edgecolor='black', label='Histogram')
    plt.plot(x_vals, kde_vals, color=color, linewidth=2, label='KDE Curve')
    
    plt.yscale('log')  # Enable log scale
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density (log scale)")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"📊 Saved {title} plot as {save_path}")

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

def compute_statistics(gt_segmentation, presegmentation, neighbors, flow, match_threshold=0.25):
    """
    Compute statistics comparing ground truth segmentation with presegmentation.
    
    Args:
        gt_segmentation (np.ndarray): Ground truth mask.
        presegmentation (np.ndarray): Pre-segmented mask.
        neighbors (np.ndarray): Neighbor connection estimates.
        flow (np.ndarray): Flow field.
        match_threshold (float): Threshold for valid matches in segment matching.
    
    Returns:
        pd.DataFrame: Statistics results.
    """
    # # Print shapes for debugging
    # print(f"GT Shape: {gt_segmentation.shape}, Preseg Shape: {presegmentation.shape}")
    # print(f"Neighbors Shape: {neighbors.shape}, Flow Shape: {flow.shape}")
    # # Print dtypes for debugging
    # print(f"GT dtype: {gt_segmentation.dtype}, Preseg dtype: {presegmentation.dtype}")
    # print(f"Neighbors dtype: {neighbors.dtype}, Flow dtype: {flow.dtype}")
    
    presegmentation=relabel_sequentially_3D(presegmentation)

    # Get unique labels
    gt_labels = np.unique(gt_segmentation)[1:]  # Remove background (0)
    preseg_labels = np.unique(presegmentation)[1:]  # Remove background (0)
    
    # Match labels
    matched_pairs = match_labels(gt_segmentation, presegmentation, method="1:N", criterion="iop", thresh=match_threshold)
    #print(matched_pairs)
    matched_gt_labels = {gt for gt, _ in matched_pairs}
    matched_preseg_labels = {pre for _, pre in matched_pairs}
    
    # Calculate unmatched labels
    unmatched_gt = len(set(gt_labels) - matched_gt_labels)
    unmatched_preseg = len(set(preseg_labels) - matched_preseg_labels)
    
    # Compute segment sizes
    gt_sizes = [np.sum(gt_segmentation == lbl) for lbl in gt_labels]
    preseg_sizes = [np.sum(presegmentation == lbl) for lbl in preseg_labels]
    
    # Compute statistics
    stats = {
        "num_gt_segments": [len(gt_labels)],
        "num_preseg_segments": [len(preseg_labels)],
        "unmatched_gt_segments": [unmatched_gt],
        "unmatched_preseg_segments": [unmatched_preseg],
        "avg_gt_size": [np.mean(gt_sizes) if gt_sizes else 0],
        "avg_preseg_size": [np.mean(preseg_sizes) if preseg_sizes else 0]
    }

    boundaries_preseg = calculateBoundaryConnection(torch.tensor(presegmentation)).cpu().numpy()
    
    matched_relabel = np.zeros_like(presegmentation)
    offset = len(preseg_labels)
    for gt, pre in matched_pairs:
        matched_relabel[presegmentation == pre] = gt + offset
    boundaries_matched = calculateBoundaryConnection(torch.tensor(matched_relabel)).cpu().numpy()
    
   
    true_boundaries=np.where(boundaries_matched) 
    false_boundaries = np.where((boundaries_preseg == 1) & (boundaries_matched == 0))

    true_neighbor_values=neighbors[true_boundaries]
    false_neighbor_values = neighbors[false_boundaries]

    true_flow_vecs=flow[:,true_boundaries[1],true_boundaries[2],true_boundaries[3]]
    false_flow_vecs=flow[:,false_boundaries[1],false_boundaries[2],false_boundaries[3]]

    coefficients = np.array([
        [-1, 0, 0],  # For value 0
        [1, 0, 0],   # For value 1
        [0, -1, 0],  # For value 2
        [0, 1, 0],   # For value 3
        [0, 0, -1],  # For value 4
        [0, 0, 1]    # For value 5
    ])

    true_flow_cos = np.sum(coefficients[true_boundaries[0]].T * true_flow_vecs,axis=0)
    false_flow_cos = np.sum(coefficients[false_boundaries[0]].T * false_flow_vecs,axis=0)

 
    return pd.DataFrame(stats),true_neighbor_values,false_neighbor_values,true_flow_cos,false_flow_cos


### 🔹 WRAPPER FUNCTIONS

def preprocess_and_cache(data_folder, output_folder, config):
    """
    Preprocess masks and store computed flows and neighbor connectivity.

    Args:
        data_folder (str): Path to the folder containing subfolders with segmented masks.
        output_folder (str): Path where precomputed data should be stored.
        config (dict): Configuration for training. Should contain:
            - "mask_name": Filename of the mask.
            - "force_recompute": Whether to overwrite existing results (default: False).

    Returns:
        dict: Paths to precomputed datasets for each subfolder.
    """
    os.makedirs(output_folder, exist_ok=True)
    subfolders = sorted(glob(os.path.join(data_folder, "*")))  # List all subdirectories
    dataset_paths = {}

    logging.info(f"Found {len(subfolders)} subfolders to process.")
    force_recompute = config.get("force_recompute", False)

    for subfolder in tqdm(subfolders):
        if not os.path.isdir(subfolder):
            continue  # Skip non-directory files

        data_name = os.path.basename(subfolder)
        sub_output_folder = os.path.join(output_folder, data_name)
        os.makedirs(sub_output_folder, exist_ok=True)

        mask_path = os.path.join(subfolder, config["mask_name"])
        cache_paths = {
            "flows": os.path.join(sub_output_folder, "flows.npy"),
            "neighbors": os.path.join(sub_output_folder, "neighbors.npy"),
        }

        # **Skip if already computed and force_recompute=False**
        if not force_recompute and all(os.path.exists(path) for path in cache_paths.values()):
            logging.info(f"Skipping {data_name}, precomputed data found.")
            dataset_paths[data_name] = cache_paths
            continue

        if not check_required_files([mask_path], data_name):
            continue

        # Compute and save results
        mask = tiff.imread(mask_path)
        np.save(cache_paths["flows"], calculateFlow(mask))
        np.save(cache_paths["neighbors"], calculateNeighborConnection(torch.tensor(mask)).cpu().numpy())

        dataset_paths[data_name] = cache_paths  

    logging.info("✅ Preprocessing completed. Data cached for all subfolders.")
    return dataset_paths

def train_and_save_model(config):
    """
    Trains the UNet3D model and saves results if no existing trained model is found.

    Args:
        config (dict): Configuration dictionary containing:
            - "data_folder": Path to the dataset.
            - "results_folder": Path for storing results.
            - "mask_name": Filename of the mask file.
            - "nuclei_name": Filename of the nuclei file.
            - "profile_name": Filename of the profile file.
            - "force_recompute": Whether to overwrite existing model checkpoints (default: False).
    """
    data_folder = config["data_folder"]
    results_folder = os.path.join(config["results_folder"], "precomputed")
    checkpoint_dir = os.path.join(config["results_folder"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # **Check if a trained model already exists**
    checkpoint_path = os.path.join(checkpoint_dir, config["model_name"])  # Assuming PyTorch model
    force_recompute = config.get("force_recompute", False)

    if not force_recompute and os.path.exists(checkpoint_path):
        logging.info(f"Skipping training. Model checkpoint already exists at {checkpoint_path}")
        return

    masks, nuclei, profiles, flows, neighbors = [], [], [], [], []

    for subfolder in sorted(os.listdir(data_folder)):
        data_path = os.path.join(data_folder, subfolder)
        result_path = os.path.join(results_folder, subfolder)

        required_files = {
            "mask": os.path.join(data_path, config["mask_name"]),
            "nuclei": os.path.join(data_path, config["nuclei_name"]),
            "profile": os.path.join(data_path, config["profile_name"]),
            "flow": os.path.join(result_path, "flows.npy"),
            "neighbors": os.path.join(result_path, "neighbors.npy"),
        }

        if not check_required_files(required_files.values(), subfolder):
            continue  # Skip this subfolder if any file is missing

        # Load and store data
        masks.append(tiff.imread(required_files["mask"]) > 0)
        nuclei.append(tiff.imread(required_files["nuclei"]))
        profiles.append(np.load(required_files["profile"]))
        flows.append(np.load(required_files["flow"]))
        neighbors.append(np.load(required_files["neighbors"]))

    # Train the model
    logging.info("Starting UNet3D training...")
    config["checkpoint_dir"] = checkpoint_dir
    train_model(config, masks, nuclei, profiles, flows, neighbors)
    logging.info(f"✅ Training complete. Model saved in {checkpoint_dir}")

def run_pre_segmentation(config):
    """
    Performs pre-segmentation by applying connected components on the mask and flow field.

    Args:
        config (dict): Configuration dictionary containing:
            - "results_folder": Base output directory.
            - "mask_name": Filename of the mask file.
            - "flow_name": Filename of the flow field file.
            - "force_recompute": Whether to overwrite existing pre-segmentation results (default: False).
    """
    applied_results_folder = os.path.join(config["results_folder"], "applied_to_gt")
    preseg_output_folder = os.path.join(config["results_folder"], "presegmentation")
    os.makedirs(preseg_output_folder, exist_ok=True)

    subfolders = sorted([f for f in os.listdir(applied_results_folder) if os.path.isdir(os.path.join(applied_results_folder, f))])
    force_recompute = config.get("force_recompute", False)

    logging.info(f"Found {len(subfolders)} subfolders for pre-segmentation.")

    for subfolder in tqdm(subfolders, desc="Pre-segmentation"):
        subfolder_path = os.path.join(applied_results_folder, subfolder)
        output_folder = os.path.join(preseg_output_folder, subfolder)
        os.makedirs(output_folder, exist_ok=True)

        # Define file paths
        mask_path = os.path.join(subfolder_path, config["mask_name"])
        flow_path = os.path.join(subfolder_path, config["flow_name"])
        preseg_output_path = os.path.join(output_folder, "presegmentation.tif")

        # **Skip computation if results exist and force_recompute=False**
        if not force_recompute and os.path.exists(preseg_output_path):
            logging.info(f"Skipping {subfolder}, pre-segmentation already exists.")
            continue

        # Check if required files exist
        if not check_required_files([mask_path, flow_path], subfolder):
            continue

        # Load mask and flow
        mask = tiff.imread(mask_path) > 0  # Convert to binary mask
        flow = np.load(flow_path)  # Shape: (3, D, H, W)

        # Compute connected components (pre-segmentation)
        labels = connected_components_3D(mask, flow)

        # Save pre-segmented labels
        tiff.imwrite(preseg_output_path, labels.astype(np.uint16))
        logging.info(f"✅ Pre-segmentation saved for {subfolder} in {preseg_output_path}")
    
    logging.info("✅ Pre-segmentation process completed for all subfolders.")

def compute_and_save_statistics(config, plot=False):
    """
    Computes and saves dataset statistics, including KDE models.

    Args:
        config (dict): Configuration dictionary containing:
            - "results_folder": Base output directory.
            - "data_folder": Path to dataset.
            - "mask_name": Filename of the mask file.
            - "force_recompute": Whether to overwrite existing statistics (default: False).
        plot (bool): Whether to generate and save KDE plots.
    """
    results_folder = os.path.join(config["results_folder"], "statistics")
    os.makedirs(results_folder, exist_ok=True)

    # **Check if statistics already exist**
    stats_file = os.path.join(results_folder, "statistics_summary.csv")
    force_recompute = config.get("force_recompute", False)

    if not force_recompute and os.path.exists(stats_file):
        logging.info(f"Skipping statistics computation. Existing file found: {stats_file}")
        return

    preseg_folder = os.path.join(config["results_folder"], "presegmentation")
    applied_to_gt_folder = os.path.join(config["results_folder"], 'applied_to_gt')

    all_data = {"true_neighbors": [], "false_neighbors": [], "true_flow_cos": [], "false_flow_cos": []}
    all_stats = []

    for subfolder in sorted(os.listdir(preseg_folder)):
        gt_path = os.path.join(config["data_folder"], subfolder, config["mask_name"])
        required_files = {
            "gt": gt_path,
            "preseg": os.path.join(preseg_folder, subfolder, "presegmentation.tif"),
            "neighbors": os.path.join(applied_to_gt_folder, subfolder, "neighbors.npy"),
            "flows": os.path.join(applied_to_gt_folder, subfolder, "flows.npy"),
        }

        if not check_required_files(required_files.values(), subfolder):
            continue  # Skip if any required file is missing

        # Load data
        gt_segmentation = tiff.imread(required_files["gt"])
        presegmentation = tiff.imread(required_files["preseg"])
        neighbors = np.load(required_files["neighbors"])
        flow = np.load(required_files["flows"])

        # Compute statistics
        stats_df, *data_values = compute_statistics(gt_segmentation, presegmentation, neighbors, flow)
        stats_df.insert(0, "dataset", subfolder)  # Add dataset name
        all_stats.append(stats_df)

        for key, data in zip(all_data.keys(), data_values):
            all_data[key].append(data)

    # Save final statistics
    if all_stats:
        final_stats = pd.concat(all_stats, ignore_index=True)
        final_stats.to_csv(stats_file, index=False)
        logging.info(f"✅ Statistics saved to {stats_file}")
    else:
        logging.warning("⚠️ No statistics were computed. Skipping KDE fitting.")
        return  # Exit early if no data was processed

    # Save KDEs and plots
    for key, color in zip(all_data.keys(), ["blue", "red", "green", "purple"]):
        kde_file = os.path.join(results_folder, f"kde_{key}.pkl")

        if not force_recompute and os.path.exists(kde_file):
            logging.info(f"Skipping KDE fitting for {key}. Existing file found: {kde_file}")
            continue

        kde = fit_and_save_kde(np.concatenate(all_data[key]), f"kde_{key}.pkl", results_folder)

        if plot:
            plot_distribution(
                np.concatenate(all_data[key]),
                kde,
                key.replace("_", " ").title(),
                color,
                os.path.join(results_folder, f"{key}.png")
            )

### 🔹 HOLE PIPELINE

def main(config):
    """
    Main pipeline to preprocess 3D data and train UNet3D.

    Args:
        config (dict): configuration.
    """
    data_folder = config["data_folder"]
    results_folder = config["results_folder"]
    os.makedirs(results_folder, exist_ok=True)

    logging.info("Starting preprocessing step...")
    preprocess_and_cache(data_folder, os.path.join(results_folder, "precomputed"),config)


    logging.info("Starting training...")
    train_and_save_model(config)
    logging.info("✅ Training complete. Model saved in results folder.")
   
    logging.info("Starting model application...")
    config["apply_result_folder"]= os.path.join(config["results_folder"],'applied_to_gt') 
    apply_model(config)

    if config.get("visualize", False):
        logging.info("Starting visualization...")
        visualize_results(config)
    
    logging.info("Starting pre-segmentation...")
    run_pre_segmentation(config)

    logging.info("Starting statistics estimation...")
    compute_and_save_statistics(config,True)
