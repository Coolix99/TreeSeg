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

    config["dataset_paths"] = dataset_paths  # Update config with structured paths

    logging.info("✅ Preprocessing completed. Data cached for all subfolders.")
    return dataset_paths

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
        train_model(config)

        logging.info("✅ Training complete. Model saved in results folder.")
   

