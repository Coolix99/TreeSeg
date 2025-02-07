import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import tifffile as tiff
from tree_seg.network_3D.pepare_data import calculateFlow, calculateNeighborConnection  
from tree_seg.network_3D.train_unet import train_model  
from tree_seg.core.pre_segmentation import connected_components_3D
from tree_seg.pipeline.apply_model3d import main as apply_model
from tree_seg.pipeline.visualize import visualize_results
from tree_seg.metrices.label_operations import relabel_sequentially_3D
import pandas as pd
from tree_seg.metrices.matching import match_labels

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_and_cache(data_folder, output_folder, config):
    """
    Preprocess ground truth masks and store computed flows and neighbor connectivity.

    Args:
        data_folder (str): Path to the folder containing subfolders with segmented masks.
        output_folder (str): Path where precomputed data should be stored.
        config (dict): Configuration for training.

    Returns:
        dict: Paths to precomputed datasets for each subfolder.
    """
    os.makedirs(output_folder, exist_ok=True)

    subfolders = sorted(glob(os.path.join(data_folder, "*")))  # List all subdirectories
    dataset_paths = {}

    logging.info(f"Found {len(subfolders)} subfolders to process.")

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

        # Skip if already computed
        if all(os.path.exists(path) for path in cache_paths.values()):
            logging.info(f"Skipping {data_name}, precomputed data found.")
            dataset_paths[data_name] = cache_paths
            continue


        mask = tiff.imread(mask_path)

        # Compute flow & neighbor connectivity
        flow = calculateFlow(mask)
        neighbors = calculateNeighborConnection(torch.tensor(mask)).cpu().numpy()


        # Save precomputed data
        np.save(cache_paths["flows"], flow)
        np.save(cache_paths["neighbors"], neighbors)

        dataset_paths[data_name] = cache_paths  # Store paths for this subfolder

    logging.info("✅ Preprocessing completed. Data cached for all subfolders.")
    return dataset_paths

def wrapper_train_model(config):
    data_folder = config["data_folder"]
    results_folder = os.path.join(config["results_folder"], "precomputed")

    mask_name = config["mask_name"]
    nuclei_name = config["nuclei_name"]
    profile_name = config["profile_name"]

    # Collect all data
    masks_list, nuclei_list, profiles_list, flows_list, neighbors_list = [], [], [], [], []

    subfolders = sorted([f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))])

    for subfolder in subfolders:
        data_path = os.path.join(data_folder, subfolder)
        result_path = os.path.join(results_folder, subfolder)

        mask_path = os.path.join(data_path, mask_name)
        nuclei_path = os.path.join(data_path, nuclei_name)
        profile_path = os.path.join(data_path, profile_name)

        flow_path = os.path.join(result_path, "flows.npy")
        neighbor_path = os.path.join(result_path, "neighbors.npy")

        if not all(os.path.exists(p) for p in [mask_path, nuclei_path, profile_path, flow_path, neighbor_path]):
            print(f"Skipping {subfolder}, missing required files.")
            continue

        # Load data
        mask = tiff.imread(mask_path)>0
        nuclei = tiff.imread(nuclei_path)
        profile = np.load(profile_path)
        flow = np.load(flow_path)
        neighbors = np.load(neighbor_path)

        # Validate profile shape matches context size
        if profile.shape[-1] != config["context_size"]:
            raise ValueError(f"Profile shape mismatch in {subfolder}: Expected {config['context_size']}, got {profile.shape[-1]}")

        # Store data
        masks_list.append(mask)
        nuclei_list.append(nuclei)
        profiles_list.append(profile)
        flows_list.append(flow)
        neighbors_list.append(neighbors)

    config["checkpoint_dir"]=os.path.join(config["results_folder"], "checkpoints")

    train_model(config, masks_list, nuclei_list, profiles_list, flows_list, neighbors_list)

def wrapper_pre_segmentation(config):
    """
    Performs pre-segmentation by applying connected components on the mask and flow field.

    Args:
        config (dict): Configuration dictionary containing input/output paths.
    """
    applied_results_folder = os.path.join(config["results_folder"], "applied_to_gt")
    preseg_output_folder = os.path.join(config["results_folder"], "presegmentation")
    os.makedirs(preseg_output_folder, exist_ok=True)

    subfolders = sorted([f for f in os.listdir(applied_results_folder) if os.path.isdir(os.path.join(applied_results_folder, f))])

    logging.info(f"Found {len(subfolders)} subfolders for pre-segmentation.")

    for subfolder in tqdm(subfolders):
        subfolder_path = os.path.join(applied_results_folder, subfolder)
        output_folder = os.path.join(preseg_output_folder, subfolder)
        os.makedirs(output_folder, exist_ok=True)

        # Define file paths
        mask_path = os.path.join(subfolder_path, config["mask_name"])
        flow_path = os.path.join(subfolder_path, config["flow_name"])
        preseg_output_path = os.path.join(output_folder, "presegmentation.tif")

        # Skip if pre-segmentation already exists
        if os.path.exists(preseg_output_path):
            logging.info(f"Skipping {subfolder}, pre-segmentation already exists.")
            continue

        # Check if required files exist
        if not os.path.exists(mask_path) or not os.path.exists(flow_path):
            logging.warning(f"Skipping {subfolder}, missing mask or flow file.")
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
     


def compute_statistics(gt_segmentation, presegmentation, neighbors, flow, match_threshold=0.3):
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
    
    return pd.DataFrame(stats)

def wrapper_compute_statistics(config):
    """
    Computes statistics for all datasets and stores the results.
    
    Args:
        config (dict): Configuration dictionary with paths.
    """
    results_folder = os.path.join(config["results_folder"], "statistics")
    os.makedirs(results_folder, exist_ok=True)

    preseg_folder = os.path.join(config["results_folder"], "presegmentation")
    data_folder = config["data_folder"]
    precomputed_folder = os.path.join(config["results_folder"], "precomputed")

    all_stats = []

    subfolders = sorted([f for f in os.listdir(preseg_folder) if os.path.isdir(os.path.join(preseg_folder, f))])
    logging.info(f"Computing statistics for {len(subfolders)} datasets.")

    for subfolder in subfolders:
        gt_path = os.path.join(data_folder, subfolder, config["mask_name"])
        preseg_path = os.path.join(preseg_folder, subfolder, "presegmentation.tif")
        neighbor_path = os.path.join(precomputed_folder, subfolder, "neighbors.npy")
        flow_path = os.path.join(precomputed_folder, subfolder, "flows.npy")

        if not all(os.path.exists(p) for p in [gt_path, preseg_path, neighbor_path, flow_path]):
            logging.warning(f"Skipping {subfolder}, missing files.")
            continue

        # Load data
        gt_segmentation = tiff.imread(gt_path)
        presegmentation = tiff.imread(preseg_path)
        neighbors = np.load(neighbor_path)
        flow = np.load(flow_path)

        # Compute statistics
        stats_df = compute_statistics(gt_segmentation, presegmentation, neighbors, flow)
        stats_df.insert(0, "dataset", subfolder)  # Add dataset name
        all_stats.append(stats_df)

       

    # Aggregate and save results
    if all_stats:
        final_stats = pd.concat(all_stats, ignore_index=True)
        stats_file = os.path.join(results_folder, "statistics_summary.csv")
        final_stats.to_csv(stats_file, index=False)
        logging.info(f"✅ Statistics saved to {stats_file}")
    else:
        logging.warning("No statistics were computed.")



def main(config, preprocess=True, train=True, apply=True, vis=False, preseg=True, stat_estimates=True):
    """
    Main pipeline to preprocess 3D data and train UNet3D.

    Args:
        config (dict): configuration.
    """
    data_folder = config["data_folder"]
    results_folder = config["results_folder"]
    os.makedirs(results_folder, exist_ok=True)

    if preprocess:
        # Preprocess and cache data
        logging.info("Starting preprocessing step...")
        preprocess_and_cache(data_folder, os.path.join(results_folder, "precomputed"),config)

    if train:
        # Train the model
        logging.info("Starting training...")
        wrapper_train_model(config)
        logging.info("✅ Training complete. Model saved in results folder.")
   
    if apply:
        # Apply the trained model
        logging.info("Starting model application...")
        config["apply_result_folder"]= os.path.join(config["results_folder"],'applied_to_gt') 
        apply_model(config)

    if vis:
        # Visualize results
        logging.info("Starting visualization...")
        visualize_results(config)
    
    if preseg:
        # Pre-segmentation
        logging.info("Starting pre-segmentation...")
        wrapper_pre_segmentation(config)

    if stat_estimates:
        # Compute statistics
        logging.info("Starting statistics estimation...")
        wrapper_compute_statistics(config)
