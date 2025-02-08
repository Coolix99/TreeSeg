import torch
import numpy as np
import higra as hg
import matplotlib.pyplot as plt
from tree_seg.metrices.label_operations import relabel_sequentially_3D
from tree_seg.core.neighbor_calculations import calculateBoundaryConnection

def construct_connection_graph(preseg_mask, flow, neighbors):
    """
    Constructs a region adjacency graph (RAG) for pre-segmentation regions.

    Args:
        preseg_mask (np.ndarray): Pre-segmented mask.
        flow (np.ndarray): Flow field (shape: (3, D, H, W)).
        neighbors (np.ndarray): Neighbor connection estimates (shape: [6, D, H, W]).

    Returns:
        hg.UndirectedGraph: The region adjacency graph (RAG) for the segmentation.
        dict: Dictionary mapping edge (segment1, segment2) -> boundary pixels.
        dict: Dictionary mapping edge (segment1, segment2) -> boundary normal directions.
        dict: Dictionary mapping edge (segment1, segment2) -> {neighbor_values, flow_cos_values}.
    """
    # Ensure labels are consecutive
    preseg_mask = relabel_sequentially_3D(preseg_mask)
    unique_labels = np.unique(preseg_mask)[1:]  # Exclude background (0)

    # Compute boundary connectivity (6-channel binary tensor)
    boundaries = calculateBoundaryConnection(torch.tensor(preseg_mask)).cpu().numpy()

    # Initialize the RAG manually
    graph = hg.UndirectedGraph(len(unique_labels))
    adjacency_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=bool)

    edge_boundaries = {}
    edge_normals = {}
    edge_values = {}

    # Define the 6 possible normal directions
    coefficients = np.array([
        [-1, 0, 0],  # z-
        [1, 0, 0],   # z+
        [0, -1, 0],  # y-
        [0, 1, 0],   # y+
        [0, 0, -1],  # x-
        [0, 0, 1]    # x+
    ])
    for direction in range(6):  # Iterate over the 6 possible boundary directions
        boundary_mask = boundaries[direction]
        boundary_pixels = np.array(np.where(boundary_mask)).T  # Shape: (N, 3) -> (z, y, x)

        for z, y, x in boundary_pixels:
            region1 = preseg_mask[z, y, x]
            #print(region1)
            shifted_z, shifted_y, shifted_x = z - coefficients[direction, 0], y - coefficients[direction, 1], x - coefficients[direction, 2]
            #print(z,y,x)
            # Ensure the neighbor is within bounds
            if 0 <= shifted_z < preseg_mask.shape[0] and 0 <= shifted_y < preseg_mask.shape[1] and 0 <= shifted_x < preseg_mask.shape[2]:
                region2 = preseg_mask[shifted_z, shifted_y, shifted_x]
                #print(region1, region2)
                if region1 != region2 and region1 > 0 and region2 > 0:  # Ignore background
                    idx1, idx2 = region1 - 1, region2 - 1  # Convert to zero-based indexing
                    #print(idx1,idx2)
                    if not adjacency_matrix[idx1, idx2]:  # Avoid duplicate edges
                        graph.add_edge(idx1, idx2)
                        adjacency_matrix[idx1, idx2] = adjacency_matrix[idx2, idx1] = True

                    # Store boundary information
                    edge_key = (int(region1), int(region2))
                    if edge_key not in edge_boundaries:
                        edge_boundaries[edge_key] = []
                        edge_normals[edge_key] = []
                        edge_values[edge_key] = {"neighbor_values": [], "flow_cos_values": []}

                    edge_boundaries[edge_key].append((z, y, x))
                    edge_normals[edge_key].append(coefficients[direction])

                    # ✅ **FIX: Correct indexing for `neighbors` using `direction`**
                    neighbor_value = neighbors[direction, z, y, x]
                    edge_values[edge_key]["neighbor_values"].append(neighbor_value)

                    # Extract corresponding flow value
                    flow_vec = flow[:, z, y, x]
                    flow_cos = np.dot(coefficients[direction], flow_vec)
                    edge_values[edge_key]["flow_cos_values"].append(flow_cos)

    # Convert lists to arrays for efficiency
    for key in edge_boundaries:
        edge_boundaries[key] = np.array(edge_boundaries[key])
        edge_normals[key] = np.array(edge_normals[key])
        edge_values[key]["neighbor_values"] = np.array(edge_values[key]["neighbor_values"])
        edge_values[key]["flow_cos_values"] = np.array(edge_values[key]["flow_cos_values"])

    return graph, edge_boundaries, edge_normals, edge_values

def construct_segmentation_graph(preseg_mask, flow, neighbors, kde_models):
    """
    Constructs a region adjacency graph (RAG) and assigns probabilities to each edge 
    using KDE-based Bayesian inference.

    Args:
        preseg_mask (np.ndarray): Pre-segmented mask.
        flow (np.ndarray): Flow field (shape: (3, D, H, W)).
        neighbors (np.ndarray): Neighbor connection estimates.
        kde_models (dict): Pre-trained KDE models for probability estimation.

    Returns:
        hg.UndirectedGraph: The region adjacency graph (RAG) with probabilities assigned.
        dict: Dictionary mapping edge (segment1, segment2) -> probability p(true).
    """
    # Step 1: Build adjacency graph and extract edge information
    graph, edge_boundaries, edge_normals, edge_values = construct_connection_graph(preseg_mask, flow, neighbors)
    
    edge_probabilities = {}  # Store P(true) for each edge
    all_probabilities = []   # Store probabilities for visualization

    # Step 2: Iterate over all edges
    for edge, values in edge_values.items():
        neighbor_values = values["neighbor_values"]
        flow_cos_values = values["flow_cos_values"]

        # Compute likelihoods using KDE models
        L_true_neighbors = kde_models["true_neighbors"].pdf(neighbor_values)
        L_false_neighbors = kde_models["false_neighbors"].pdf(neighbor_values)

        L_true_flow_cos = kde_models["true_flow_cos"].pdf(flow_cos_values)
        L_false_flow_cos = kde_models["false_flow_cos"].pdf(flow_cos_values)

        # Multiply along the boundary pixels
        L_true = np.prod(L_true_neighbors) * np.prod(L_true_flow_cos)
        L_false = np.prod(L_false_neighbors) * np.prod(L_false_flow_cos)

        # Normalize probabilities to ensure p(true) + p(false) = 1
        P_true = L_true / (L_true + L_false)
        
        edge_probabilities[edge] = P_true
        all_probabilities.append(P_true)

    # Step 3: Plot histogram of P(true) distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_probabilities, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("P(True) for Segment Connections")
    plt.ylabel("Density")
    plt.title("Histogram of Edge Probabilities")
    plt.grid(True)
    plt.show()

    return graph, edge_probabilities
