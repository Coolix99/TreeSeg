import os
import json
import logging
import numpy as np
import torch
import tifffile as tiff
from tqdm import tqdm
from glob import glob
from tree_seg.network_3D.apply_unet import apply_model  # Import apply_model function

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def apply_model_to_folders(data_folder, results_folder, config):
    """
    Apply the trained UNet3D model to all subfolders in the dataset.

    Args:
        data_folder (str): Path to the folder containing subfolders with input images.
        results_folder (str): Path where processed results should be stored.
        config (dict): Configuration for applying the model.
    """
    os.makedirs(results_folder, exist_ok=True)

    subfolders = sorted(glob(os.path.join(data_folder, "*")))  # List all subdirectories

    logging.info(f"Found {len(subfolders)} subfolders to process.")

    for subfolder in tqdm(subfolders):
        if not os.path.isdir(subfolder):
            continue  # Skip non-directory files

        data_name = os.path.basename(subfolder)
        sub_output_folder = os.path.join(results_folder, data_name)
        os.makedirs(sub_output_folder, exist_ok=True)

        # Define input and output paths
        image_path = os.path.join(subfolder, config["nuclei_name"])
        seg_output_path = os.path.join(sub_output_folder, "segmentation.tif")
        flow_output_path = os.path.join(sub_output_folder, "pred_flows.npy")
        neighbor_output_path = os.path.join(sub_output_folder, "neighbor_preds.npy")

        # Skip processing if segmentation already exists
        if os.path.exists(seg_output_path):
            logging.info(f"Skipping {data_name}, results already exist.")
            continue

        # Load input image
        if not os.path.exists(image_path):
            logging.warning(f"Skipping {data_name}, missing image file: {image_path}")
            continue

        image = tiff.imread(image_path)
        profile_path = os.path.join(subfolder, config["profile_name"])
        if not os.path.exists(profile_path):
            logging.warning(f"Skipping {data_name}, missing profile file: {profile_path}")
            continue
        profile=np.load(profile_path)

        # Apply model
        logging.info(f"Processing {data_name}...")
        segmentation, pred_flows, neighbor_preds = apply_model(config, image,profile)

        # Save results
        tiff.imwrite(seg_output_path, segmentation.astype(np.uint8))
        np.save(flow_output_path, pred_flows)
        np.save(neighbor_output_path, neighbor_preds)

        logging.info(f"✅ Processed {data_name}: Saved results in {sub_output_folder}")

def main(config):
    """
    Main pipeline to apply the trained model to 3D images.

    Args:
        config (dict): Configuration settings.
    """
    data_folder = config["data_folder"]
    results_folder = os.path.join(config["results_folder"], "applied")
    os.makedirs(results_folder, exist_ok=True)

    logging.info("Starting model application...")
    apply_model_to_folders(data_folder, results_folder, config)
    logging.info("✅ Model application complete. Results saved.")

