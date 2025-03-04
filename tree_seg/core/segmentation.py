import torch
import numpy as np
import higra as hg
import matplotlib.pyplot as plt

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

    # determine presegment sizes
    segment_sizes = {label: np.sum(preseg_mask == label) for label in unique_labels}

    return graph, edge_boundaries, edge_normals, edge_values,segment_sizes

def construct_segmentation_graph(preseg_mask, flow, neighbors, l_models):
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
    # Build adjacency graph and extract edge information
    graph, edge_boundaries, edge_normals, edge_values,segment_sizes = construct_connection_graph(preseg_mask, flow, neighbors)
    
    edge_probabilities = {}  # Store P(true) for each edge
    all_probabilities = []   # Store probabilities for visualization


    # Iterate over all edges
    for edge, values in edge_values.items():
        neighbor_values = values["neighbor_values"]
        flow_cos_values = values["flow_cos_values"]

        l_true_neighbors = l_models["true_neighbors"](neighbor_values)
        l_false_neighbors = l_models["false_neighbors"](neighbor_values)
        l_true_flow_cos = l_models["true_flow_cos"](flow_cos_values)
        l_false_flow_cos = l_models["false_flow_cos"](flow_cos_values)

        region1, region2 = edge
        print('edge',edge)
        vol_min = np.log(min(segment_sizes[region1], segment_sizes[region2])/100)
        print('vol_min',vol_min)
        vol_max = np.log(max(segment_sizes[region1], segment_sizes[region2])/100)
        l_true_vol_min = l_models["true_boundary_log100_volumes_min"](vol_min)
        l_true_vol_max = l_models["true_boundary_log100_volumes_max"](vol_max)
        l_false_vol_min = l_models["false_boundary_log100_volumes_min"](vol_min)
        l_false_vol_max = l_models["false_boundary_log100_volumes_max"](vol_max)

        # print(L_true_neighbors.max(),L_true_neighbors.min())
        # print(L_false_neighbors.max(),L_false_neighbors.min())
        # print(L_true_flow_cos.max(),L_true_flow_cos.min())
        # print(L_false_flow_cos.max(),L_false_flow_cos.min())
        # print('---------')

        # l_true = np.sum(l_true_neighbors) + np.sum(l_true_flow_cos)
        # l_false = np.sum(l_false_neighbors) + np.sum(l_false_flow_cos)

        l_true = np.sum(l_true_neighbors) + np.sum(l_true_flow_cos) + l_true_vol_min + l_true_vol_max
        l_false = np.sum(l_false_neighbors) + np.sum(l_false_flow_cos) + l_false_vol_min+ + l_false_vol_max

        # l_true = l_true_vol_min + l_true_vol_max 
        # l_false = l_false_vol_min+ l_false_vol_max

        L_true = np.exp(l_true)
        L_false = np.exp(l_false)
        P_true = L_true / (L_true + L_false)
        #print(l_true,l_false,P_true)
        edge_probabilities[edge] = P_true
        all_probabilities.append(P_true)

    # Plot histogram of P(true) distribution
    plt.figure(figsize=(8, 5))
    plt.hist(all_probabilities, bins=500, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("P(True) for Segment Connections")
    plt.ylabel("Density")
    plt.title("Histogram of Edge Probabilities")
    plt.grid(True)
    plt.show()

    return graph, edge_probabilities, edge_values,segment_sizes


def construct_threshold_segmentation(preseg_mask, graph, edge_probabilities, threshold=0.5):
    """
    Merges segments in `preseg_mask` based on edge probabilities in `graph`.

    Args:
        preseg_mask (np.ndarray): Pre-segmented mask.
        graph (hg.UndirectedGraph): Region adjacency graph (RAG).
        edge_probabilities (dict): Dictionary mapping edge (segment1, segment2) -> probability p(true).
        threshold (float): Probability threshold for merging segments.

    Returns:
        np.ndarray: New segmented image after merging.
    """
    # Get unique labels
    unique_labels = np.unique(preseg_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

    # Initialize Union-Find (Disjoint Set)
    label_to_index = {int(label): int(i) for i, label in enumerate(unique_labels)}
    parent = np.arange(len(unique_labels))  # Each segment is its own parent initially

    def find(x):
        """Find with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return int(parent[x])

    def union(x, y):
        """Union by rank."""
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Merge segments with `p(true) <= threshold`
    i=0
    for edge, prob in edge_probabilities.items():
        if prob <= threshold:  # Merge segments
            region1, region2 = edge
            if region1 in label_to_index and region2 in label_to_index:
                i+=1
                idx1, idx2 = int(label_to_index[region1]), int(label_to_index[region2])
                union(idx1, idx2)
    print('merge',i)
    # Create a mapping from old labels to new merged labels
    new_labels = {int(label): find(label_to_index[label]) + 1 for label in unique_labels}  # +1 to avoid 0
    #  Relabel the preseg_mask
    threshold_segmentation = np.vectorize(lambda x: new_labels.get(x, 0))(preseg_mask)

    return threshold_segmentation

def update_probabilities(edge_probabilities, merged_region1, merged_region2, graph, l_models, edge_values, segment_sizes):
    """
    Updates edge probabilities after merging two segments.

    Args:
        edge_probabilities (dict): Dictionary mapping edge (segment1, segment2) -> probability p(true).
        merged_region1 (int): The surviving region after merging.
        merged_region2 (int): The absorbed region.
        graph (hg.UndirectedGraph): Region adjacency graph (RAG).
        l_models (dict): Pre-trained KDE models for probability estimation.
        edge_values (dict): Edge feature values (neighbor_values, flow_cos_values).
        segment_sizes (dict): Dictionary mapping segment labels to their sizes.

    Returns:
        dict: Updated edge probabilities.
    """
    affected_edges = set()

    # Find edges that involve either merged_region1 or merged_region2
    for edge in list(edge_probabilities.keys()):
        region1, region2 = edge
        if merged_region2 in edge:  # If region2 was the merged one, update it to region1
            new_edge = (merged_region1, region1) if region2 == merged_region2 else (merged_region1, region2)
            affected_edges.add(new_edge)
            del edge_probabilities[edge]  # Remove the old edge

    # Update segment sizes
    segment_sizes[merged_region1] += segment_sizes[merged_region2]
    del segment_sizes[merged_region2]  # Remove old segment size

    # Recalculate probabilities for affected edges
    for edge in affected_edges:
        region1, region2 = edge
        if edge in edge_values:
            neighbor_values = edge_values[edge]["neighbor_values"]
            flow_cos_values = edge_values[edge]["flow_cos_values"]

            vol_min = min(segment_sizes[region1], segment_sizes[region2])
            vol_max = max(segment_sizes[region1], segment_sizes[region2])

            l_true_neighbors = l_models["true_neighbors"](neighbor_values)
            l_false_neighbors = l_models["false_neighbors"](neighbor_values)
            l_true_flow_cos = l_models["true_flow_cos"](flow_cos_values)
            l_false_flow_cos = l_models["false_flow_cos"](flow_cos_values)

            l_true_vol_min = l_models["true_boundary_log100_volumes_min"](vol_min)
            l_true_vol_max = l_models["true_boundary_log100_volumes_max"](vol_max)
            l_false_vol_min = l_models["false_boundary_log100_volumes_min"](vol_min)
            l_false_vol_max = l_models["false_boundary_log100_volumes_max"](vol_max)

            l_true = np.sum(l_true_neighbors) + np.sum(l_true_flow_cos) + l_true_vol_min + l_true_vol_max
            l_false = np.sum(l_false_neighbors) + np.sum(l_false_flow_cos) + l_false_vol_min + l_false_vol_max

            L_true = np.exp(l_true)
            L_false = np.exp(l_false)
            P_true = L_true / (L_true + L_false)

            edge_probabilities[edge] = P_true  # Store updated probability

    return edge_probabilities

def construct_adaptive_segmentation(preseg_mask, graph, edge_probabilities, l_models, edge_values, segment_sizes, threshold=0.5):
    """
    Iteratively merges segments in `preseg_mask` based on edge probabilities, dynamically updating after each merge.

    Args:
        preseg_mask (np.ndarray): Pre-segmented mask.
        graph (hg.UndirectedGraph): Region adjacency graph (RAG).
        edge_probabilities (dict): Dictionary mapping edge (segment1, segment2) -> probability p(true).
        threshold (float): Stopping criterion; merging stops when the smallest probability > threshold.

    Returns:
        np.ndarray: New segmented image after merging.
    """
    # Get unique labels
    unique_labels = np.unique(preseg_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

    # Initialize Union-Find (Disjoint Set)
    label_to_index = {int(label): int(i) for i, label in enumerate(unique_labels)}
    parent = np.arange(len(unique_labels))  # Each segment is its own parent initially

    def find(x):
        """Find with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return int(parent[x])

    def union(x, y):
        """Union by rank."""
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Sort edges by probability (ascending order)
    sorted_edges = sorted(edge_probabilities.items(), key=lambda x: x[1])

    merged_count = 0

    while sorted_edges:
        # Pick the lowest probability edge
        (region1, region2), prob = sorted_edges.pop(0)

        if prob > threshold:
            break  # Stop merging when the probability exceeds threshold

        # Merge segments
        if region1 in label_to_index and region2 in label_to_index:
            idx1, idx2 = int(label_to_index[region1]), int(label_to_index[region2])
            union(idx1, idx2)
            merged_count += 1

            # Update edge probabilities after merging
            edge_probabilities = update_probabilities(edge_probabilities, region1, region2, graph, l_models, edge_values, segment_sizes)

            # Re-sort edges after updating probabilities
            sorted_edges = sorted(edge_probabilities.items(), key=lambda x: x[1])

    print('Total merged:', merged_count)

    # Create a mapping from old labels to new merged labels
    new_labels = {int(label): find(label_to_index[label]) + 1 for label in unique_labels}  # +1 to avoid 0
    # Relabel the preseg_mask
    adaptive_segmentation = np.vectorize(lambda x: new_labels.get(x, 0))(preseg_mask)

    return adaptive_segmentation
