import numpy as np

from typing import Any
import opytimizer.utils.exception as e


def weighted_sum(fit: np.ndarray, weights: np.ndarray, xp: Any = np,**kwargs) -> float:
    """The weighted sum classical method.
    Args:
        fit: agent fit value.
        weights: weights vector
    """
   
    return xp.sum((fit * weights), axis=-1)


def tchebycheff(fit: np.ndarray, weights: np.ndarray, z: np.ndarray, xp: Any = np,**kwargs) -> float:
    """The Tchebycheff classical method.
    Args:
        fit: agent fit value.
        weights: weights vector
        z: reference point.
    """
    
    return xp.max(weights * (xp.abs(fit - z)), axis=-1)


def pbi(
    fit: np.ndarray, weights: np.ndarray, z: np.ndarray, penalty: float = 5.0, xp: Any = np,**kwargs
) -> float:
    """The Penalty-based Boundary Intersection (PBI) method.

    Args:
        fit: Objective values of the agent (agent.fit).
        weights: Weight vector (should be normalized).
        z: Reference point.
        penalty: Penalty parameter (typically a large positive number, e.g., 5.0 or 10.0).

    Returns:
        (float): Scalar value computed using the PBI method.
    """
    # Normalize weight vector
    
    norm_w = xp.linalg.norm(weights)
    if norm_w == 0:
        raise e.ValueError("Weight vector must not be zero.")
    w = weights / norm_w

    # Compute the difference vector
    diff = fit - z

    # Projection (d1): length along the direction of weight vector
    d1 = xp.sum(diff * w, axis=-1)

    if fit.ndim == 2:
        proj = d1[:, xp.newaxis] * w
    else:
        proj = d1 * w

    d2 = xp.linalg.norm(diff - proj, axis=-1)

    # Final PBI value
    return d1 + penalty * d2
