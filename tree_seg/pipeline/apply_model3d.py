import os
import logging
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from glob import glob
from tree_seg.network_3D.apply_unet import apply_model  

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
    force_recompute = config.get("force_recompute", False)

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
        mask_output_path = os.path.join(sub_output_folder, config["mask_name"])
        flow_output_path = os.path.join(sub_output_folder, config["flow_name"])
        neighbor_output_path = os.path.join(sub_output_folder, config["neighbor_name"])

        # Skip processing if segmentation already exists
        if os.path.exists(mask_output_path) and not force_recompute:
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
        pred_mask, pred_flow, pred_neighbors = apply_model(config, image,profile)
        
        # Save results
        tiff.imwrite(mask_output_path, pred_mask.astype(bool))
        np.save(flow_output_path, pred_flow)
        np.save(neighbor_output_path, pred_neighbors)

        logging.info(f"✅ Processed {data_name}: Saved results in {sub_output_folder}")

def main(config):
    """
    Main pipeline to apply the trained model to 3D images.

    Args:
        config (dict): Configuration settings.
    """
    data_folder = config["data_folder"]
    results_folder = config["apply_result_folder"]
    os.makedirs(results_folder, exist_ok=True)

    logging.info("Starting model application...")
    apply_model_to_folders(data_folder, results_folder, config)
    logging.info("✅ Model application complete. Results saved.")

