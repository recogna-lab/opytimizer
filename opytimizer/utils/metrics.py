import numpy as np


def _ensure_2d(arr):
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def calculate_hypervolume(
    pareto_front: np.ndarray, reference_point: np.ndarray
) -> float:
    """
    Calculates the Hypervolume (HV) of a set of solutions.
    Args:
        pareto_front (np.ndarray): Array of solutions (n_solutions, n_objectives).
        reference_point (np.ndarray): Reference point (n_objectives,).
    Returns:
        float: Hypervolume value.
    """
    pareto_front = _ensure_2d(pareto_front)
    if pareto_front.shape[0] < 2:
        return 0.0
    sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
    hv = 0.0
    prev = reference_point.copy()
    for point in reversed(sorted_front):
        width = prev[0] - point[0]
        height = reference_point[1] - point[1]
        hv += width * height
        prev[1] = point[1]
    return hv


def calculate_igd(pareto_front: np.ndarray, pareto_optimal: np.ndarray) -> float:
    """
    Calculates the Inverted Generational Distance (IGD).
    Args:
        pareto_front (np.ndarray): Obtained solutions (n_solutions, n_objectives).
        pareto_optimal (np.ndarray): True Pareto front (n_optimal, n_objectives).
    Returns:
        float: IGD value.
    """
    pareto_front = _ensure_2d(pareto_front)
    pareto_optimal = _ensure_2d(pareto_optimal)
    if pareto_front.shape[0] == 0 or pareto_optimal.shape[0] == 0:
        return 0.0
    distances = []
    for optimal in pareto_optimal:
        d = np.linalg.norm(pareto_front - optimal, axis=1)
        distances.append(np.min(d))
    return np.mean(distances)


def calculate_gd(pareto_front: np.ndarray, pareto_optimal: np.ndarray) -> float:
    """
    Calculates the Generational Distance (GD).
    Args:
        pareto_front (np.ndarray): Obtained solutions.
        pareto_optimal (np.ndarray): True Pareto front.
    Returns:
        float: GD value.
    """
    pareto_front = _ensure_2d(pareto_front)
    pareto_optimal = _ensure_2d(pareto_optimal)
    if pareto_front.shape[0] == 0 or pareto_optimal.shape[0] == 0:
        return 0.0
    distances = []
    for solution in pareto_front:
        d = np.linalg.norm(pareto_optimal - solution, axis=1)
        distances.append(np.min(d))
    return np.mean(distances)


def calculate_spread(pareto_front: np.ndarray, pareto_optimal: np.ndarray) -> float:
    """
    Calculates the Spread (Delta) of a set of solutions.
    Args:
        pareto_front (np.ndarray): Obtained solutions.
        pareto_optimal (np.ndarray): True Pareto front.
    Returns:
        float: Spread value.
    """
    pareto_front = _ensure_2d(pareto_front)
    pareto_optimal = _ensure_2d(pareto_optimal)
    if pareto_front.shape[0] < 2 or pareto_optimal.shape[0] == 0:
        return 0.0
    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
    df = np.linalg.norm(pareto_front[1:] - pareto_front[:-1], axis=1)
    d_mean = np.mean(df)
    d_f = np.min(np.linalg.norm(pareto_front[0] - pareto_optimal, axis=1))
    d_l = np.min(np.linalg.norm(pareto_front[-1] - pareto_optimal, axis=1))
    delta = (d_f + d_l + np.sum(np.abs(df - d_mean))) / (d_f + d_l + (len(df)) * d_mean)
    return delta


def calculate_error_ratio(
    pareto_front: np.ndarray, pareto_optimal: np.ndarray, tol: float = 1e-6
) -> float:
    """
    Calculates the Error Ratio (ER).
    Args:
        pareto_front (np.ndarray): Obtained solutions.
        pareto_optimal (np.ndarray): True Pareto front.
        tol (float): Tolerance to consider a solution as belonging to the front.
    Returns:
        float: Error Ratio value.
    """
    pareto_front = _ensure_2d(pareto_front)
    pareto_optimal = _ensure_2d(pareto_optimal)
    if pareto_front.shape[0] == 0 or pareto_optimal.shape[0] == 0:
        return 0.0
    errors = 0
    for solution in pareto_front:
        distances = np.linalg.norm(pareto_optimal - solution, axis=1)
        if np.min(distances) > tol:
            errors += 1
    return errors / len(pareto_front) if len(pareto_front) > 0 else 0.0


def calculate_r2(pareto_front: np.ndarray, weight_vectors: np.ndarray) -> float:
    """
    Calculates the R2-metric.
    Args:
        pareto_front (np.ndarray): Obtained solutions.
        weight_vectors (np.ndarray): Weight vectors (n_weights, n_objectives).
    Returns:
        float: R2-metric value.
    """
    pareto_front = _ensure_2d(pareto_front)
    weight_vectors = _ensure_2d(weight_vectors)
    if pareto_front.shape[0] == 0 or weight_vectors.shape[0] == 0:
        return 0.0
    r2_values = []
    for w in weight_vectors:
        tchebycheff = np.max(np.abs((pareto_front) * w), axis=1)
        r2_values.append(np.min(tchebycheff))
    return np.mean(r2_values)


def calculate_maximum_spread(pareto_front: np.ndarray) -> float:
    """
    Calculates the Maximum Spread (MS).
    Args:
        pareto_front (np.ndarray): Obtained solutions.
    Returns:
        float: Maximum Spread value.
    """
    pareto_front = _ensure_2d(pareto_front)
    if pareto_front.shape[0] < 2:
        return 0.0
    dists = np.linalg.norm(pareto_front[None, :, :] - pareto_front[:, None, :], axis=2)
    return np.max(dists)
