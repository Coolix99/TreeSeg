import os
import numpy as np
import tifffile as tiff
import napari
import logging
from glob import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import numpy as np

def convert_flow_to_vectors(flow):
    """
    Convert a dense 3D flow field into a Napari-compatible vector field format.

    Args:
        flow (numpy.ndarray): Shape (3, D, H, W), representing (dz, dy, dx) at each voxel.

    Returns:
        vectors (numpy.ndarray): Shape (N, 2, 3), where:
            - First column represents vector start (X, Y, Z) coordinates.
            - Second column represents vector direction (dx, dy, dz).
    """
    D, H, W = flow.shape[1:]  # Extract spatial dimensions

    # Create a grid of coordinates in (X, Y, Z) order
    X, Y, Z = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing="ij")

    # Flatten the coordinate grid to list all voxel positions
    start_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # Shape (N, 3)

    # Flatten the flow vectors and reorder components correctly
    U = flow[2].ravel()  # dx (X component)
    V = flow[1].ravel()  # dy (Y component)
    W = flow[0].ravel()  # dz (Z component)

    # Compute vector directions
    direction = np.vstack([W, V, U]).T  # Shape (N, 3)

    # Combine start points and directions into Napari vector format
    vectors = np.stack([start_points,  direction], axis=1)  # Shape (N, 2, 3)

    return vectors



def visualize_results(config):
    """
    Iterate through all processed results and visualize them in Napari.

    Args:
        config (dict): Configuration dictionary containing paths and visualization options.
    """
    data_folder = config["data_folder"]
    app_gt_subfolder = os.path.join(config['results_folder'], 'applied_to_gt') 
    precomputed_subfolder = os.path.join(config['results_folder'], 'precomputed') 

    subfolders = sorted(glob(os.path.join(data_folder, "*")))  # List all subdirectories

    logging.info(f"Found {len(subfolders)} subfolders to visualize.")

    for subfolder in tqdm(subfolders):
        if not os.path.isdir(subfolder):
            continue  # Skip non-directory files

        data_name = os.path.basename(subfolder)
        app_gt_folder = os.path.join(app_gt_subfolder, data_name)
        precomputed_folder = os.path.join(precomputed_subfolder, data_name)

        image_path = os.path.join(subfolder, config["nuclei_name"])
        gt_seg_path = os.path.join(subfolder, config["mask_name"])
        flow_gt_path = os.path.join(precomputed_folder, config["flow_name"])
        neighbor_gt_path = os.path.join(precomputed_folder, config["neighbor_name"])
        seg_output_path = os.path.join(app_gt_folder, config["mask_name"])
        flow_output_path = os.path.join(app_gt_folder, config["flow_name"])
        neighbor_output_path = os.path.join(app_gt_folder, config["neighbor_name"])

        # Load data
        nuclei = tiff.imread(image_path)
        gt_seg = tiff.imread(gt_seg_path)
        gt_flow = np.load(flow_gt_path)
        gt_neighbors = np.load(neighbor_gt_path)
        segmentation = tiff.imread(seg_output_path)
        flow = np.load(flow_output_path)
        neighbors = np.load(neighbor_output_path)

        

        # Create Napari viewer
        viewer = napari.Viewer()

        # Display nuclei image
        viewer.add_image(nuclei, name="Nuclei", colormap="gray", blending="additive")

        # Display segmentation results
        viewer.add_labels(segmentation, name="Segmentation")


        for i in range(6):
            viewer.add_image(neighbors[i], name=f"Neighbor {i+1}", colormap="viridis")

        # Display ground truth segmentation
        viewer.add_labels(gt_seg, name="GT Segmentation")

  
        for i in range(6):
            viewer.add_labels(gt_neighbors[i], name=f"GT Neighbor {i+1}")

        # Display flow vectors as dense arrow fields
        if config.get("show_flow", False):
            # Convert flow fields to vector format for Napari
            gt_vectors = convert_flow_to_vectors(gt_flow)
            pred_vectors = convert_flow_to_vectors(flow)
            viewer.add_vectors(pred_vectors, name="Predicted Flow", edge_color="red", edge_width=0.2, length=1)
            viewer.add_vectors(gt_vectors, name="GT Flow", edge_color="blue", edge_width=0.2, length=1)

        napari.run()
