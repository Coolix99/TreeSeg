import numpy as np
import pytest
from tree_seg.metrices.utils import is_array_of_integers, label_are_sequential, _check_label_array

def test_is_array_of_integers():
    assert is_array_of_integers(np.array([1, 2, 3]))
    assert not is_array_of_integers(np.array([1.0, 2.0, 3.0]))
    assert not is_array_of_integers([1, 2, 3])  # not a numpy array

def test_label_are_sequential():
    assert label_are_sequential(np.array([1, 2, 3]))
    assert not label_are_sequential(np.array([1, 3, 4]))

def test_check_label_array():
    # Valid cases
    _check_label_array(np.array([0, 1, 2, 3]))
    _check_label_array(np.array([1, 2, 3]), check_sequential=True)

    # Invalid cases
    with pytest.raises(ValueError):
        _check_label_array(np.array([-1, 0, 1]))  # Negative integer
    with pytest.raises(ValueError):
        _check_label_array(np.array([1, 3, 4]), check_sequential=True)  # Non-sequential
    with pytest.raises(ValueError):
        _check_label_array(np.array([1.0, 2.0, 3.0]))  # Not an integer array
