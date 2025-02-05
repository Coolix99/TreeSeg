import os
import json
import logging
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import tifffile as tiff
from tree_seg.network_3D.pepare_data import calculateFlow, calculateNeighborConnection  
from tree_seg.network_3D.train_unet import train_model  

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

def main(config, preprocess=True, train=True):
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
   

