import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import combinations

# from bokeh.plotting import figure, show
# from bokeh.layouts import gridplot
# from bokeh.models import  HoverTool

from tree_seg.network_3D.pepare_data import calculateFlow, calculateNeighborConnection  
from tree_seg.network_3D.train_unet import train_model  
from tree_seg.core.pre_segmentation import connected_components_3D, euler_connected_components_3D
from tree_seg.pipeline.apply_model3d import main as apply_model
from tree_seg.pipeline.visualize import visualize_traininig_res
from tree_seg.metrices.label_operations import relabel_sequentially_3D
from tree_seg.metrices.matching import match_labels
from tree_seg.core.neighbor_calculations import calculateBoundaryConnection
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

def save_concatenated_data(concatenated_data, output_folder):
    """
    Saves the concatenated true/false data as CSV files.

    Args:
        concatenated_data (dict): Dictionary containing the combined arrays.
        output_folder (str): Directory to save the CSV files.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Separate true and false keys
    true_keys = [key for key in concatenated_data.keys() if key.startswith("true_")]
    false_keys = [key for key in concatenated_data.keys() if key.startswith("false_")]

    # Create DataFrames
    df_true = pd.DataFrame({key.replace("true_", ""): concatenated_data[key] for key in true_keys})
    df_false = pd.DataFrame({key.replace("false_", ""): concatenated_data[key] for key in false_keys})

    # Save to CSV
    true_csv_path = os.path.join(output_folder, "true_data.csv")
    false_csv_path = os.path.join(output_folder, "false_data.csv")

    df_true.to_csv(true_csv_path, index=False)
    df_false.to_csv(false_csv_path, index=False)

    print(f"Saved: {true_csv_path}")
    print(f"Saved: {false_csv_path}")

def plot_distribution(data, title, color, save_path):
    """Plots histogram and KDE curve, saves the plot."""
    if len(data) == 0 :
        logging.warning(f"Skipping {title}, missing data ")
        return
    
    # x_vals = np.linspace(min(data), max(data), 200)
    # kde_vals = kde(x_vals)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.5, color=color, edgecolor='black', label='Histogram')
    # plt.plot(x_vals, kde_vals, color=color, linewidth=2, label='KDE Curve')
    
    plt.yscale('log')  # Enable log scale
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density (log scale)")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"📊 Saved {title} plot as {save_path}")

def compute_kde_contour(x, y, grid_size=100, levels=5):
    """
    Compute 2D Kernel Density Estimate (KDE) and contour levels.
    
    Args:
        x, y (np.ndarray): 1D arrays of data points.
        grid_size (int): Resolution of KDE grid.
        levels (int): Number of contour levels.

    Returns:
        xx, yy (meshgrid): Grid coordinates.
        density (np.ndarray): KDE density values on grid.
        contour_levels (list): Contour density values.
    """
    # Grid range
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))

    # Fit KDE
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kde = stats.gaussian_kde(values)(positions).reshape(xx.shape)

    # Compute contour levels
    contour_levels = np.linspace(kde.min(), kde.max(), levels)

    return xx, yy, kde, contour_levels

def sample_data(data, sample_size=200):
    """Randomly samples rows while keeping column alignment."""
    if len(data) > sample_size:
        idx = np.random.choice(data.shape[0], sample_size, replace=False)
        return data[idx]
    return data

def plot_corner_contour(concatenated_data, sample_size=5000, contour_levels=10, outlier_threshold=0.05):
    """
    Creates a corner plot using KDE contours for dense regions 
    and scatter points for outliers using Bokeh.

    Args:
        concatenated_data (dict): Dictionary with 1D numpy arrays for each key.
        sample_size (int): Number of points to sample for efficiency.
        contour_levels (int): Number of contour levels.
        outlier_threshold (float): Threshold for outlier detection.
    """
    var_names = ["Neighbors", "Flow Cos", "Boundary Volume"]

    # Stack data first, then sample
    true_data = np.column_stack([
        concatenated_data["true_neighbors"], 
        concatenated_data["true_flow_cos"], 
        concatenated_data["true_boundary_log100_volumes_max"]
    ])
    
    false_data = np.column_stack([
        concatenated_data["false_neighbors"], 
        concatenated_data["false_flow_cos"], 
        concatenated_data["false_boundary_log100_volumes_max"]
    ])

    # Sample rows while preserving alignment
    true_data = sample_data(true_data, sample_size)
    false_data = sample_data(false_data, sample_size)

    df_true = pd.DataFrame(true_data, columns=var_names)
    df_false = pd.DataFrame(false_data, columns=var_names)

    plots = []
    for x_var, y_var in combinations(var_names, 2):  
        x_true, y_true = df_true[x_var].values, df_true[y_var].values
        x_false, y_false = df_false[x_var].values, df_false[y_var].values

        # Compute KDE for contours
        xx, yy, kde_true, levels_true = compute_kde_contour(x_true, y_true, levels=contour_levels)
        xx, yy, kde_false, levels_false = compute_kde_contour(x_false, y_false, levels=contour_levels)

        # Identify outliers based on KDE threshold
        kde_true_eval = stats.gaussian_kde(np.vstack([x_true, y_true]))(np.vstack([x_true, y_true]))
        kde_false_eval = stats.gaussian_kde(np.vstack([x_false, y_false]))(np.vstack([x_false, y_false]))

        outliers_true = kde_true_eval < np.percentile(kde_true_eval, outlier_threshold * 100)
        outliers_false = kde_false_eval < np.percentile(kde_false_eval, outlier_threshold * 100)

        # Create Bokeh plot
        p = figure(title=f"{x_var} vs {y_var}", tools="pan,wheel_zoom,box_zoom,reset,save")

        # Contours for dense regions
        p.contour(xx, yy, kde_true, levels=levels_true, fill_alpha=0.3,line_color=["blue"]*5 )
        p.contour(xx, yy, kde_false, levels=levels_false, fill_alpha=0.3,line_color=["red"]*5 )

        # Scatter for outliers
        p.scatter(x=x_true[outliers_true], y=y_true[outliers_true], fill_color="blue", alpha=0.5, legend_label="True Outliers")
        p.scatter(x=x_false[outliers_false], y=y_false[outliers_false], fill_color="red", alpha=0.5, legend_label="False Outliers")

        # Tooltips
        hover = HoverTool(tooltips=[(x_var, f"@{x_var}"), (y_var, f"@{y_var}")])
        p.add_tools(hover)

        p.legend.location = "top_right"
        p.xaxis.axis_label = x_var
        p.yaxis.axis_label = y_var
        p.grid.grid_line_alpha = 0.3
        plots.append(p)

    # Arrange plots in a grid
    grid = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)])  # 2 per row
    show(grid)

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
    preseg_sizes = {int(lbl):np.sum(presegmentation == lbl) for lbl in preseg_labels}
    # Compute statistics
    stats = {
        "num_gt_segments": [len(gt_labels)],
        "num_preseg_segments": [len(preseg_labels)],
        "unmatched_gt_segments": [unmatched_gt],
        "unmatched_preseg_segments": [unmatched_preseg],
        #"avg_gt_size": [np.mean(gt_sizes) if gt_sizes else 0],
        #"avg_preseg_size": [np.mean(preseg_sizes) if preseg_sizes else 0]
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

    def get_boundary_volumes(boundaries):
        segment1 = presegmentation[boundaries[1], boundaries[2], boundaries[3]]
        segment2 = presegmentation[boundaries[1] - coefficients[boundaries[0], 0],
                                   boundaries[2] - coefficients[boundaries[0], 1],
                                   boundaries[3] - coefficients[boundaries[0], 2]]
        # print(boundaries)
        # print(segment1)
        # print(segment2)    
        #volumes = np.array([preseg_sizes[int(s1)] + preseg_sizes[int(s2)] for s1, s2 in zip(segment1, segment2)])
        log100_volumes_min = np.log(np.array([min(preseg_sizes[int(s1)], preseg_sizes[int(s2)]) for s1, s2 in zip(segment1, segment2)])/100)
        log100_volumes_max = np.log(np.array([max(preseg_sizes[int(s1)], preseg_sizes[int(s2)]) for s1, s2 in zip(segment1, segment2)])/100)
        #volumes_abs_diff = np.array([abs(preseg_sizes[int(s1)] - preseg_sizes[int(s2)]) for s1, s2 in zip(segment1, segment2)])

        
        return log100_volumes_min,log100_volumes_max

    true_boundary_log100_volumes_min,true_boundary_log100_volumes_max = get_boundary_volumes(true_boundaries)
    false_boundary_log100_volumes_min,false_boundary_log100_volumes_max = get_boundary_volumes(false_boundaries)
    # print(false_boundaries[0].shape)
    # print(false_boundary_volumes.shape)
    
    return pd.DataFrame(stats),true_neighbor_values,false_neighbor_values, \
            true_flow_cos,false_flow_cos, \
            true_boundary_log100_volumes_min,true_boundary_log100_volumes_max, \
            false_boundary_log100_volumes_min,false_boundary_log100_volumes_max

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

        gt_segmentation_path = os.path.join(subfolder, config["gt_segmentation_name"])
        cache_paths = {
            "gt_flows": os.path.join(sub_output_folder, config["gt_flow_name"]),
            "gt_neighbors": os.path.join(sub_output_folder, config["gt_neighbor_name"]),
        }

        # **Skip if already computed and force_recompute=False**
        if not force_recompute and all(os.path.exists(path) for path in cache_paths.values()):
            logging.info(f"Skipping {data_name}, precomputed data found.")
            dataset_paths[data_name] = cache_paths
            continue

        if not check_required_files([gt_segmentation_path], data_name):
            continue

        # Compute and save results
        gt_segmentation = tiff.imread(gt_segmentation_path)
        np.save(cache_paths["gt_flows"], calculateFlow(gt_segmentation))
        np.save(cache_paths["gt_neighbors"], calculateNeighborConnection(torch.tensor(gt_segmentation)).cpu().numpy())

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

    gt_segmentation, nuclei, profiles, gt_flows, gt_neighbors = [], [], [], [], []

    for subfolder in sorted(os.listdir(data_folder)):
        data_path = os.path.join(data_folder, subfolder)
        result_path = os.path.join(results_folder, subfolder)

        required_files = {
            "gt_segmentation": os.path.join(data_path, config["gt_segmentation_name"]),
            "nuclei": os.path.join(data_path, config["gt_nuclei_name"]),
            "profile": os.path.join(data_path, config["gt_profile_name"]),
            "gt_flow": os.path.join(result_path, config["gt_flow_name"]),
            "gt_neighbors": os.path.join(result_path, config["gt_neighbor_name"]),
        }

        if not check_required_files(required_files.values(), subfolder):
            continue  # Skip this subfolder if any file is missing

        # Load and store data
        gt_segmentation.append(tiff.imread(required_files["gt_segmentation"]) > 0)
        nuclei.append(tiff.imread(required_files["nuclei"]))
        profiles.append(np.load(required_files["profile"]))
        gt_flows.append(np.load(required_files["gt_flow"]))
        gt_neighbors.append(np.load(required_files["gt_neighbors"]))

    # Train the model
    logging.info("Starting UNet3D training...")
    config["checkpoint_dir"] = checkpoint_dir
    train_model(config, gt_segmentation, nuclei, profiles, gt_flows, gt_neighbors)
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
        gt_segmentation_path = os.path.join(subfolder_path, config["mask_name"])
        flow_path = os.path.join(subfolder_path, config["flow_name"])
        preseg_output_path = os.path.join(output_folder, "presegmentation.tif")

        # **Skip computation if results exist and force_recompute=False**
        if not force_recompute and os.path.exists(preseg_output_path):
            logging.info(f"Skipping {subfolder}, pre-segmentation already exists.")
            continue

        # Check if required files exist
        if not check_required_files([gt_segmentation_path, flow_path], subfolder):
            continue

        # Load mask and flow
        mask = tiff.imread(gt_segmentation_path) > 0  # Convert to binary mask
        flow = np.load(flow_path)  # Shape: (3, D, H, W)

        # Compute connected components (pre-segmentation)
        #labels = connected_components_3D(mask, flow)
        labels = euler_connected_components_3D(mask, flow)

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

    all_data = {"true_neighbors": [], "false_neighbors": [], 
                "true_flow_cos": [], "false_flow_cos": [], 
                "true_boundary_log100_volumes_min":[],"true_boundary_log100_volumes_max":[],
                "false_boundary_log100_volumes_min":[],"false_boundary_log100_volumes_max":[]}
    all_stats = []

    for subfolder in sorted(os.listdir(preseg_folder)):
        gt_path = os.path.join(config["data_folder"], subfolder, config["gt_segmentation_name"])
        required_files = {
            "gt": gt_path,
            "preseg": os.path.join(preseg_folder, subfolder, "presegmentation.tif"),
            "neighbors": os.path.join(applied_to_gt_folder, subfolder, config["neighbor_name"]),
            "flows": os.path.join(applied_to_gt_folder, subfolder, config["flow_name"]),
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

    # Save concatenated data and plots
    concatenated_data={}
    for key, color in zip(all_data.keys(), ["blue", "red", "green", "purple", "black", "black", "black", "black"]):
        #kde_file = os.path.join(results_folder, f"kde_{key}.pkl")

        concatenated_data[key]=np.concatenate(all_data[key])
        #kde = fit_and_save_kde(concatenated_data[key], f"kde_{key}.pkl", results_folder)
        #print(key,np.mean(concatenated_data[key]))
        if plot:
            plot_distribution(
                concatenated_data[key],
                key.replace("_", " ").title(),
                color,
                os.path.join(results_folder, f"{key}.png")
            )
    save_concatenated_data(concatenated_data, results_folder)
    #plot_corner_contour(concatenated_data)
    
### 🔹 HOLE PIPELINE

def train_main(config):
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
    config["apply_result_folder"] = os.path.join(config["results_folder"],'applied_to_gt') 
    config["model_path"] = os.path.join(config["results_folder"],"checkpoints",config["model_name"])
    config["nuclei_name"] = config["gt_nuclei_name"]
    config["profile_name"] = config["gt_profile_name"]
    apply_model(config)

    logging.info("Starting pre-segmentation...")
    run_pre_segmentation(config)
    
    logging.info("Starting statistics estimation...")
    compute_and_save_statistics(config,True)
    
    if config.get("visual_debugging", False):
        logging.info("Starting visualization...")
        visualize_traininig_res(config)

