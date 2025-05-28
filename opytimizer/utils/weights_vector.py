import numpy as np

def ref_dirs(n_objectives: int, n_partitions: int) -> tuple[np.ndarray,int]:
    """
    Generates reference direction vectors (weight vectors) on the unit simplex.

    Parameters:
    - n_objectives: int, dimensionality of the objective space
    - n_partitions: int, number of divisions (the higher, the more vectors)

    Returns:
    - ref_dirs: np.ndarray of shape (N, n_objectives), where each row is a weight vector
    - n_subproblems: an integer that corresponds to the number of subproblems (each weight is related with a subproblem)
    """
    grid = np.mgrid[(slice(0, n_partitions+1),) * (n_objectives-1)]
    grid = grid.reshape(n_objectives-1, -1).T
    
    # Filter points where sum <= H
    mask = np.sum(grid, axis=1) <= n_partitions
    grid = grid[mask]
    
    # Compute last element and normalize
    last_elem = n_partitions - np.sum(grid, axis=1, keepdims=True)
    weights = np.hstack([grid, last_elem]) / n_partitions
    
    # Remove duplicates
    unique_weights = np.unique(weights, axis=0)
    
    return unique_weights,unique_weights.shape[0]