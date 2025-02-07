# tests/test_intersection.py
import numpy as np
import pytest
from tree_seg.metrices.intersection import intersection_over_union, intersection_over_true, intersection_over_pred

def test_intersection_over_union():
    overlap = np.array([[4, 0], [0, 2]])
    expected_iou = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = intersection_over_union(overlap)
    assert np.allclose(result, expected_iou), "IoU calculation is incorrect."

def test_intersection_over_true():
    overlap = np.array([[4, 0], [0, 2]])
    expected_iot = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = intersection_over_true(overlap)
    assert np.allclose(result, expected_iot), "IoT calculation is incorrect."

def test_intersection_over_pred():
    overlap = np.array([[4, 0], [0, 2]])
    expected_iop = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = intersection_over_pred(overlap)
    assert np.allclose(result, expected_iop), "IoP calculation is incorrect."

if __name__ == "__main__":
    pytest.main()
