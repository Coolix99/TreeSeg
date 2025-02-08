import os
import logging
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import pickle
from scipy.interpolate import interp1d

from tree_seg.core.pre_segmentation import connected_components_3D
from tree_seg.pipeline.apply_model3d import main as apply_model
from tree_seg.pipeline.visualize import visualize_results
from tree_seg.core.segmentation import construct_segmentation_graph,construct_threshold_segmentation
from tree_seg.metrices.label_operations import relabel_sequentially_3D

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

def kde_to_interpolator(kde, num_points=200):
    """Precompute KDE values and create an interpolated function for fast lookup."""
    x_grid = np.linspace(kde.dataset.min(), kde.dataset.max(), num_points)
    kde_values = kde.pdf(x_grid)

    # Create an interpolator
    interpolator = interp1d(x_grid, kde_values, kind="linear", fill_value=(kde_values[0], kde_values[-1]), bounds_error=False)
    return interpolator


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

def load_and_prepare_segmentation(config):
    """
    Loads pre-segmentation results, flow, neighbor connectivity, and KDEs for final segmentation.

    Args:
        config (dict): Configuration dictionary containing:
            - "results_folder": Base output directory.
            - "force_recompute": Whether to overwrite existing results (default: False).
    
    Returns:
        dict: Dictionary containing loaded data.
    """
    results_folder = config["results_folder"]
    preseg_folder = os.path.join(results_folder, "presegmentation_data")
    applied_folder = os.path.join(results_folder, "applied")
    statistics_folder = os.path.join(results_folder, "statistics")
    final_segmentation_folder = os.path.join(results_folder, "final_segmentation")
    
    os.makedirs(final_segmentation_folder, exist_ok=True)
    force_recompute = config.get("force_recompute", False)

    # Load KDE models
    kde_paths = {
        "true_neighbors": os.path.join(statistics_folder, "kde_true_neighbors.pkl"),
        "false_neighbors": os.path.join(statistics_folder, "kde_false_neighbors.pkl"),
        "true_flow_cos": os.path.join(statistics_folder, "kde_true_flow_cos.pkl"),
        "false_flow_cos": os.path.join(statistics_folder, "kde_false_flow_cos.pkl"),
    }

    kde_models = {}
    for key, path in kde_paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                kde_models[key] = kde_to_interpolator(pickle.load(f))
            logging.info(f"✅ Loaded KDE model: {key}")
        else:
            logging.warning(f"⚠️ Missing KDE model: {key}, some logic may not work as expected.")

    subfolders = sorted([f for f in os.listdir(preseg_folder) if os.path.isdir(os.path.join(preseg_folder, f))])
    logging.info(f"Found {len(subfolders)} subfolders for final segmentation.")

    for subfolder in subfolders:
        preseg_path = os.path.join(preseg_folder, subfolder, "presegmentation.tif")
        flow_path = os.path.join(applied_folder, subfolder, "flows.npy")
        neighbors_path = os.path.join(applied_folder, subfolder, "neighbors.npy")
        final_output_path = os.path.join(final_segmentation_folder, f"{subfolder}_final.tif")

        # **Skip computation if results exist and force_recompute=False**
        if not force_recompute and os.path.exists(final_output_path):
            logging.info(f"Skipping {subfolder}, final segmentation already exists.")
            continue

        if not check_required_files([preseg_path, flow_path, neighbors_path], subfolder):
            continue

        # Load pre-segmentation mask
        preseg_mask = tiff.imread(preseg_path)

        # Load flow and neighbors
        flow = np.load(flow_path)
        neighbors = np.load(neighbors_path)

        

        preseg_mask = relabel_sequentially_3D(preseg_mask)
        graph, edge_probabilities=construct_segmentation_graph(preseg_mask.copy(),flow,neighbors,kde_models)
        threshold_segmentation_01=construct_threshold_segmentation(preseg_mask.copy(),graph, edge_probabilities,0.01)
        threshold_segmentation_05=construct_threshold_segmentation(preseg_mask.copy(),graph, edge_probabilities,0.5)
        
        import napari
        viewer=napari.Viewer()
        viewer.add_labels(threshold_segmentation_05,name='thr_seg 05')
        viewer.add_labels(threshold_segmentation_01,name='thr_seg 01')
        viewer.add_labels(preseg_mask,name='preseg')
        napari.run()


    logging.info("✅ Segmentation data loaded and prepared.")






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

    logging.info("Processing final segmentation...")
    load_and_prepare_segmentation(config)

    logging.info("✅ Segmentation pipeline completed!")


