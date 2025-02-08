import torch
import numpy as np
import higra as hg
import pytest
from scipy.stats import gaussian_kde
from tree_seg.core.segmentation import construct_connection_graph, construct_segmentation_graph
from tree_seg.core.neighbor_calculations import calculateBoundaryConnection

def generate_test_data(shape=(5, 5, 5)):
    """
    Generates synthetic segmentation, flow, and neighbor data for testing.

    Returns:
        preseg_mask (np.ndarray): Synthetic pre-segmented mask.
        flow (np.ndarray): Random flow field (3, D, H, W).
        neighbors (np.ndarray): Randomized neighbor values (6, D, H, W).
    """
    preseg_mask = np.zeros(shape, dtype=np.uint8)
    
    # Create three labeled regions
    preseg_mask[1:3, 1:3, 1:3] = 1
    preseg_mask[3:5, 1:3, 1:3] = 2
    preseg_mask[1:3, 3:5, 1:3] = 3

    # Random flow field
    flow = np.random.rand(3, *shape).astype(np.float32) * 2 - 1  # Values between -1 and 1

    # Randomized neighbor values
    neighbors = np.random.rand(6, *shape).astype(np.float32)

    return preseg_mask, flow, neighbors

def generate_test_data_minimal(shape=(5, 5, 5)):
    """
    Generates synthetic segmentation, flow, and neighbor data for testing.

    Returns:
        preseg_mask (np.ndarray): Synthetic pre-segmented mask.
        flow (np.ndarray): Random flow field (3, D, H, W).
        neighbors (np.ndarray): Randomized neighbor values (6, D, H, W).
    """
    preseg_mask = np.zeros(shape, dtype=np.uint8)
    
    # Create three labeled regions
    preseg_mask[2, 1, 2] = 1
    preseg_mask[2, 2, 2] = 1
    preseg_mask[2, 3, 2] = 2
    preseg_mask[2, 3, 3] = 3

    # Random flow field
    flow = np.random.rand(3, *shape).astype(np.float32) * 2 - 1  # Values between -1 and 1

    # Randomized neighbor values
    neighbors = np.random.rand(6, *shape).astype(np.float32)

    return preseg_mask, flow, neighbors

def test_construct_connection_graph():
    """
    Tests the construction of the region adjacency graph (RAG).
    Ensures proper connections between adjacent regions.
    """
    preseg_mask, flow, neighbors = generate_test_data()

    graph, edge_boundaries, edge_normals, edge_values = construct_connection_graph(preseg_mask, flow, neighbors)

    assert isinstance(graph, hg.UndirectedGraph)
    assert graph.num_vertices() > 0  # Ensure at least some segments exist
    assert graph.num_edges() > 0  # Ensure at least some connections exist


def test_construct_connection_graph_result():
    """
    Tests the construction of the region adjacency graph (RAG).
    Ensures proper connections between adjacent regions.
    """
    preseg_mask, flow, neighbors = generate_test_data_minimal()

    graph, edge_boundaries, edge_normals, edge_values = construct_connection_graph(preseg_mask, flow, neighbors)

    assert isinstance(graph, hg.UndirectedGraph)
    assert graph.num_vertices() > 0  # Ensure at least some segments exist
    assert graph.num_edges() > 0  # Ensure at least some connections exist

    expected={
        (1, 2): np.array([[2, 2, 2]]),
        (2, 1): np.array([[2, 3, 2]]),
        (2, 3): np.array([[2, 3, 2]]), 
        (3, 2): np.array([[2, 3, 3]])
    }

    for key in expected:
        assert key in edge_boundaries, f"Missing edge {key} in edge_boundaries"
        assert np.array_equal(edge_boundaries[key], expected[key]), f"Mismatch for edge {key}"



def test_construct_segmentation_graph():
    """
    Tests the full segmentation graph construction with KDE-based probability estimation.
    Ensures probability computation is stable and within valid range.
    """
    preseg_mask, flow, neighbors = generate_test_data()

    # Create synthetic KDE models
    kde_models = {
        "true_neighbors": gaussian_kde(np.random.rand(1000)),
        "false_neighbors": gaussian_kde(np.random.rand(1000)),
        "true_flow_cos": gaussian_kde(np.random.rand(1000)),
        "false_flow_cos": gaussian_kde(np.random.rand(1000))
    }

    graph, edge_probabilities = construct_segmentation_graph(preseg_mask, flow, neighbors, kde_models)

    assert isinstance(graph, hg.UndirectedGraph)
    assert graph.num_vertices() > 0  # Ensure at least some segments exist
    assert graph.num_edges() > 0  # Ensure at least some connections exist
    assert len(edge_probabilities) > 0  # Ensure probability mapping is not empty

    # Ensure probabilities are in the valid range
    for p in edge_probabilities.values():
        assert 0.0 <= p <= 1.0, "Probability must be in the range [0,1]"

if __name__ == "__main__":
    pytest.main()
