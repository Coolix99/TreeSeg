import os
import numpy as np
import tifffile as tiff
import napari
import logging
from glob import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def visualize_results(config):
    """
    Iterate through all processed results and visualize them in Napari.

    Args:
        show_flow (bool): Whether to show flow vectors.
    """
    data_folder = config["data_folder"]
    app_gt_subfolder = os.path.join(config['results_folder'],'applied_to_gt') 
    precomputed_subfolder = os.path.join(config['results_folder'],'precomputed') 

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
        viewer.add_image(nuclei, name=f"Nuclei", colormap="gray", blending="additive")

        # Display segmentation
        viewer.add_labels(segmentation, name=f"Segmentation")

        # Display flow vectors if enabled
        # if config['show_flow'] and flow is not None:
        #     flow_magnitude = np.linalg.norm(flow, axis=0)
        #     viewer.add_image(flow_magnitude, name=f"{subfolder} - Flow Magnitude", colormap="magma")

        # Display neighbor predictions (split into 6 labels)
        if neighbors is not None:
            for i in range(6):
                viewer.add_image(neighbors[i], name=f" Neighbor {i+1}", colormap="viridis")

       
        viewer.add_labels(gt_seg, name=f" GT Segmentation")
        if neighbors is not None:
            for i in range(6):
                viewer.add_labels(gt_neighbors[i], name=f"GT Neighbor {i+1}")    
        

        napari.run()
