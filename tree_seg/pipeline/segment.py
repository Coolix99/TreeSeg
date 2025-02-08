import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import tifffile as tiff
import pandas as pd

from tree_seg.network_3D.pepare_data import calculateFlow, calculateNeighborConnection  
from tree_seg.core.pre_segmentation import connected_components_3D
from tree_seg.pipeline.apply_model3d import main as apply_model
from tree_seg.pipeline.visualize import visualize_results

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

### 🔹 INFERENCE PIPELINE FUNCTIONS

def run_pre_segmentation(config):
    """
    Performs pre-segmentation using connected components and saves results.

    Args:
        config (dict): Configuration dictionary containing:
            - "results_folder": Base output directory.
            - "force_recompute": Whether to overwrite existing results (default: False).
    """
    apply_results_folder = os.path.join(config["results_folder"], config["apply_result_folder"])
    preseg_output_folder = os.path.join(config["results_folder"], "presegmentation_data")
    os.makedirs(preseg_output_folder, exist_ok=True)

    subfolders = sorted([f for f in os.listdir(apply_results_folder) if os.path.isdir(os.path.join(apply_results_folder, f))])
    force_recompute = config.get("force_recompute", False)

    logging.info(f"Found {len(subfolders)} subfolders for pre-segmentation.")

    for subfolder in tqdm(subfolders, desc="Pre-segmentation"):
        subfolder_path = os.path.join(apply_results_folder, subfolder)
        output_folder = os.path.join(preseg_output_folder, subfolder)
        os.makedirs(output_folder, exist_ok=True)

        # Define file paths
        mask_path = os.path.join(subfolder_path, config["mask_name"])
        flow_path = os.path.join(subfolder_path, "flows.npy")
        preseg_output_path = os.path.join(output_folder, "presegmentation.tif")

        # **Skip computation if results exist and force_recompute=False**
        if not force_recompute and os.path.exists(preseg_output_path):
            logging.info(f"Skipping {subfolder}, pre-segmentation already exists.")
            continue

        if not check_required_files([mask_path,flow_path], subfolder):
            continue

        # Load segmentation mask and flow
        mask = tiff.imread(mask_path)
        flow = np.load(flow_path)

        # Compute connected components (pre-segmentation)
        labels = connected_components_3D(mask, flow)

        # Save pre-segmented labels
        tiff.imwrite(preseg_output_path, labels.astype(np.uint16))
        logging.info(f"✅ Pre-segmentation saved for {subfolder} in {preseg_output_path}")

    logging.info("✅ Pre-segmentation process completed for all subfolders.")


### 🔹 HOLE PIPELINE

def main(config):
    """
    Main pipeline for applying the trained UNet3D model to unseen nuclei data.

    Args:
        config (dict): Configuration dictionary containing:
            - "nuclei_folder": Path to input nuclei images.
            - "results_folder": Base output directory.
            - "force_recompute": Whether to overwrite existing results (default: False).
            - "visualize": Whether to visualize results.
    """
    results_folder = config["results_folder"]
    os.makedirs(results_folder, exist_ok=True)

    logging.info("Starting segmentation application...")
    config["apply_result_folder"]= os.path.join(config["results_folder"],'applied') 
    apply_model(config)
   
    logging.info("Starting pre-segmentation...")
    run_pre_segmentation(config)

    if config.get("visualize", False):
        logging.info("Starting visualization...")
        visualize_results(config)

    

    logging.info("✅ Segmentation pipeline completed!")


