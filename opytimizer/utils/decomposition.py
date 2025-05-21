import numpy as np

def weighted_sum(agent: np.ndarray, weights: np.ndarray, **kwargs) -> float:
    """The weighted sum classical method.
    Args:
        agent: agent f_value.
        weights: weights vector
    """
    return np.sum((agent * weights))

def tchebycheff(agent: np.ndarray, weights: np.ndarray, z: np.ndarray) -> float:
    """The Tchebycheff classical method.
    Args:
        agent: agent f_value.
        weights: weights vector
        z: reference point.
    """
    return np.max(weights * (np.abs(agent - z))) 