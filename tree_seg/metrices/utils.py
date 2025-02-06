import numpy as np

def is_array_of_integers(y):
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)

def label_are_sequential(y):
    """Returns true if y has only sequential labels from 1..."""
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))

def _check_label_array(y, name=None, check_sequential=False):
    """Validate that the array y is of non-negative integers and optionally sequential."""
    label_type = 'sequential ' if check_sequential else ''
    err_msg = f"{'Labels' if name is None else name} must be an array of {label_type}non-negative integers."
    
    if not is_array_of_integers(y):
        raise ValueError(err_msg)
    if len(y) == 0:
        return True
    if check_sequential and not label_are_sequential(y):
        raise ValueError(err_msg)
    elif y.min() < 0:
        raise ValueError(err_msg)
    return True

def _safe_divide(numerator, denominator, eps=1e-10):
    """
    Safely divide two numbers or arrays, returning 0 where the denominator is zero.

    Parameters
    ----------
    numerator : float or ndarray
        The numerator in the division.
    denominator : float or ndarray
        The denominator in the division.
    eps : float, optional
        A small epsilon to avoid division by zero (default is 1e-10).

    Returns
    -------
    float or ndarray
        The result of numerator / denominator, or 0 where the denominator is zero.
    """
    if np.isscalar(denominator):
        return numerator / denominator if abs(denominator) > eps else 0.0
    else:
        result = np.zeros_like(denominator, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=np.abs(denominator) > eps)
        return result