import os
import logging
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import pickle
import pandas as pd
from scipy.interpolate import interp1d

from tree_seg.core.pre_segmentation import connected_components_3D
from tree_seg.pipeline.apply_model3d import main as apply_model
from tree_seg.pipeline.visualize import visualize_results
from tree_seg.core.segmentation import construct_segmentation_graph,construct_threshold_segmentation,construct_adaptive_segmentation
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

### 🔹 HISTOGRAM BINS & INTERPOLATION

def histogram_interpolator(data, bins=50, min_entries=10):
    """
    Creates an interpolator based on histogram binning.
    - Ensures at least `min_entries` per bin by adjusting bins dynamically.
    - Interpolates missing bins to maintain continuity.

    Args:
        data (np.ndarray): 1D array of values.
        bins (int): Initial number of bins.
        min_entries (int): Minimum number of samples per bin.

    Returns:
        function: Interpolated function for density estimation.
    """
    if len(data) == 0:
        return lambda x: 0  # Return zero function if no data

    # Compute initial histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    counts, _ = np.histogram(data, bins=bins)  # Count number of points per bin

    # # Ensure bins have at least `min_entries`
    # while any(counts < min_entries):
    #     bins = max(5, bins - 5)  # Reduce bin count dynamically
    #     hist, bin_edges = np.histogram(data, bins=bins, density=True)
    #     counts, _ = np.histogram(data, bins=bins)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fill empty bins by linear interpolation
    valid = counts >= min_entries
    bin_centers = bin_centers[valid]
    hist = hist[valid]

    log_hist = np.log(hist)

    # Set boundary values
    log_min = log_hist[0]  # Left boundary value
    log_max = log_hist[-1] # Right boundary value

    return interp1d(bin_centers, log_hist, kind="linear", fill_value=(log_min, log_max), bounds_error=False)


### 🔹 LOAD CSV DATA & CREATE INTERPOLATORS

def load_data_and_create_interpolators(statistics_folder):
    """
    Loads segmentation statistics from CSV and generates histogram-based interpolators.

    Args:
        statistics_folder (str): Folder containing CSV files.

    Returns:
        dict: Interpolators for each segmentation metric.
    """
    # Define paths
    true_data_path = os.path.join(statistics_folder, "true_data.csv")
    false_data_path = os.path.join(statistics_folder, "false_data.csv")

    # Load CSV data
    if not os.path.exists(true_data_path) or not os.path.exists(false_data_path):
        logging.error("Missing segmentation statistics CSV files!")
        return {}

    df_true = pd.read_csv(true_data_path)
    df_false = pd.read_csv(false_data_path)

    # Generate histogram interpolators
    interpolators = {}
    for column in df_true.columns:
        interpolators[f"true_{column}"] = histogram_interpolator(df_true[column].dropna().values)
    for column in df_false.columns:
        interpolators[f"false_{column}"] = histogram_interpolator(df_false[column].dropna().values)

    logging.info(f"✅ Loaded & processed segmentation statistics with histogram binning.")

    import matplotlib.pyplot as plt
    num_cols = len(df_true.columns)
    fig, axes = plt.subplots(num_cols, 2, figsize=(10, num_cols * 4))

    for i, column in enumerate(df_true.columns):
        # Plot true values
        ax_true = axes[i, 0]
        true_values = df_true[column].dropna().values
        x_true = np.linspace(true_values.min()-0.2, true_values.max()+0.2, 200)
        y_true = interpolators[f"true_{column}"](x_true)

        ax_true.hist(true_values, bins=30, density=True, alpha=0.5, color="blue", label="True Histogram")
        ax_true.plot(x_true, np.exp(y_true), color="black", label="True Interpolation")
        ax_true.set_title(f"True: {column}")
        ax_true.set_yscale('log')
        ax_true.legend()

        # Plot false values
        ax_false = axes[i, 1]
        false_values = df_false[column].dropna().values
        x_false = np.linspace(false_values.min()-0.2, false_values.max()+0.2, 200)
        y_false = interpolators[f"false_{column}"](x_false)

        ax_false.hist(false_values, bins=30, density=True, alpha=0.5, color="red", label="False Histogram")
        ax_false.plot(x_false, np.exp(y_false), color="black", label="False Interpolation")
        ax_false.set_title(f"False: {column}")
        ax_false.set_yscale('log')
        ax_false.legend()

    plt.tight_layout()
    plt.show()

    return interpolators

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

    # Load histogram-based interpolators
    interpolators = load_data_and_create_interpolators(statistics_folder)
    if not interpolators:
        logging.error("❌ Failed to load segmentation statistics.")
        return

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
        graph, edge_probabilities, edge_values,segment_sizes=construct_segmentation_graph(preseg_mask.copy(),flow,neighbors,interpolators)
        #threshold_segmentation_01=construct_threshold_segmentation(preseg_mask.copy(),graph, edge_probabilities,0.01)
        threshold_segmentation_05=construct_threshold_segmentation(preseg_mask.copy(),graph, edge_probabilities,0.5)
        
        adaptive_seg=construct_adaptive_segmentation(preseg_mask, graph, edge_probabilities,interpolators, edge_values, segment_sizes, threshold=0.5)
        #todo proper graph like approach

        import napari
        viewer=napari.Viewer()
        viewer.add_labels(threshold_segmentation_05,name='thr_seg 05')
        #viewer.add_labels(threshold_segmentation_01,name='thr_seg 01')
        viewer.add_labels(adaptive_seg,name='adaptive_seg')
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


