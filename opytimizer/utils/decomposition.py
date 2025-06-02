import numpy as np
import opytimizer.utils.exception as e
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

def pbi(agent: np.ndarray, weights: np.ndarray, z: np.ndarray, penalty: float = 5.0) -> float:
    """The Penalty-based Boundary Intersection (PBI) method.

    Args:
        agent: Objective values of the agent (f_value).
        weights: Weight vector (should be normalized).
        z: Reference point.
        penalty: Penalty parameter (typically a large positive number, e.g., 5.0 or 10.0).

    Returns:
        (float): Scalar value computed using the PBI method.
    """
    # Normalize weight vector
    norm_w = np.linalg.norm(weights)
    if norm_w == 0:
        raise e.ValueError("Weight vector must not be zero.")
    w = weights / norm_w

    # Compute the difference vector
    diff = agent - z

    # Projection (d1): length along the direction of weight vector
    d1 = np.dot(diff, w)

    # Orthogonal distance (d2): perpendicular to the weight vector
    proj = d1 * w
    d2 = np.linalg.norm(diff - proj)

    # Final PBI value
    return d1 + penalty * d2