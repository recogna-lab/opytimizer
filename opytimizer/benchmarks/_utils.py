"""Shared helpers of the benchmark problems (not part of the public API)."""

import math
from typing import Any

import numpy as np

from opytimizer.utils.reference_vectors import das_dennis

# Upper bound on the number of booleans held by each temporary of the blocked
# non-dominance check
_DOMINANCE_BLOCK_ELEMENTS = 1 << 24

# Fixed seed so that sampled reference fronts are reproducible across calls
_SAMPLING_SEED = 0

# Samples drawn per requested point before filtering a front with dominated parts
_OVERSAMPLING = 10


def _product_shape(p: Any, q: Any, xp: Any) -> Any:
    """Product-based shape shared by DTLZ, WFG and ZCAT.

    For M objectives and M - 1 factors per solution, it builds

        h_1 = p_1 * ... * p_{M-1}
        h_m = p_1 * ... * p_{M-m} * q_{M-m+1},   m = 2, ..., M

    so that, e.g., (p, q) = (cos, sin) yields the DTLZ2 sphere and
    (p, q) = (x, 1 - x) yields the DTLZ1 hyperplane.

    Args:
        p: Product factors of shape (N, M - 1).
        q: Closing factors of shape (N, M - 1).
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Shape values of shape (N, M).

    """

    ones = xp.ones((p.shape[0], 1), dtype=p.dtype)

    # prefix[:, t] = p_1 * ... * p_t (1-based), with prefix[:, 0] = 1
    prefix = xp.concatenate((ones, xp.cumprod(p, axis=1)), axis=1)
    closing = xp.concatenate((ones, q[:, ::-1]), axis=1)

    return prefix[:, ::-1] * closing


def _simplex_lattice(n_objectives: int, n_points: int) -> np.ndarray:
    """Das-Dennis points on the unit simplex, with at most `n_points` points.

    Args:
        n_objectives: Dimension of the simplex.
        n_points: Maximum number of points.

    Returns:
        (np.ndarray): Points of shape (P, M) whose rows sum to one.

    """

    # Largest number of partitions whose lattice still fits in `n_points`
    n_partitions = 1
    while math.comb(n_partitions + n_objectives, n_objectives - 1) <= n_points:
        n_partitions += 1

    weights, _ = das_dennis(n_objectives, n_partitions)

    return weights


def _unit_sphere(n_objectives: int, n_points: int) -> np.ndarray:
    """Points on the positive orthant of the unit hypersphere.

    Args:
        n_objectives: Number of objectives.
        n_points: Maximum number of points.

    Returns:
        (np.ndarray): Points of shape (P, M) with unit Euclidean norm.

    """

    weights = _simplex_lattice(n_objectives, n_points)

    return weights / np.linalg.norm(weights, axis=1, keepdims=True)


def _unit_grid(n_dimensions: int, n_points: int) -> np.ndarray:
    """Samples of the unit hypercube, used to map position parameters onto a front.

    A regular grid is used while it has at most four times `n_points` points;
    beyond that (many objectives), the hypercube is sampled uniformly at random
    with a fixed seed, together with its vertices.

    Args:
        n_dimensions: Dimension of the hypercube.
        n_points: Desired number of points.

    Returns:
        (np.ndarray): Samples of shape (P, n_dimensions) in [0, 1].

    """

    if n_dimensions == 1:
        return np.linspace(0.0, 1.0, max(2, n_points))[:, None]

    per_axis = max(2, int(round(n_points ** (1.0 / n_dimensions))))
    if per_axis**n_dimensions <= 4 * n_points:
        axes = [np.linspace(0.0, 1.0, per_axis)] * n_dimensions
        mesh = np.meshgrid(*axes, indexing="ij")

        return np.stack([m.ravel() for m in mesh], axis=1)

    rng = np.random.default_rng(_SAMPLING_SEED)
    vertices = np.stack(
        np.meshgrid(*([np.array([0.0, 1.0])] * n_dimensions), indexing="ij"), axis=-1
    ).reshape(-1, n_dimensions)

    return np.concatenate((vertices, rng.random((n_points, n_dimensions))), axis=0)


def _dominated_by(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Mask of the rows of `C` dominated by at least one row of `A`."""
    if A.shape[0] == 0:
        return np.zeros(C.shape[0], dtype=bool)

    no_worse = np.all(A[None, :, :] <= C[:, None, :], axis=2)
    better = np.any(A[None, :, :] < C[:, None, :], axis=2)

    return np.any(no_worse & better, axis=1)


def _non_dominated(F: np.ndarray) -> np.ndarray:
    """Removes duplicated and dominated points (minimization).

    After a lexicographic sort, a dominating point always precedes the points it
    dominates, so each point only has to be compared with the non-dominated
    points found before it.

    Args:
        F: Objective values of shape (P, M).

    Returns:
        (np.ndarray): Non-dominated rows of `F`, sorted lexicographically.

    """

    F = np.unique(F, axis=0)
    n, m = F.shape

    if m == 2:
        # Sorted by f_1: a point survives iff its f_2 beats every previous one
        previous = np.minimum.accumulate(np.concatenate(([np.inf], F[:-1, 1])))
        return F[F[:, 1] < previous]

    archive = F[:0]
    start = 0
    while start < n:
        # Block size keeps the (block, archive, M) temporaries bounded
        size = max(
            1, min(1024, _DOMINANCE_BLOCK_ELEMENTS // max(1, archive.shape[0] * m))
        )
        C = F[start : start + size]
        dominated = _dominated_by(archive, C) | _dominated_by(C, C)
        archive = np.concatenate((archive, C[~dominated]), axis=0)
        start += size

    return archive


def _farthest_points(F: np.ndarray, n_points: int) -> np.ndarray:
    """Selects `n_points` well-spread rows of `F` (greedy farthest-point sampling).

    The extreme point of each objective is kept first; then, the point farthest
    from the selected ones (in the range-normalized space) is added iteratively.

    Args:
        F: Points of shape (P, M).
        n_points: Number of points to keep.

    Returns:
        (np.ndarray): Selected rows of `F`, sorted lexicographically.

    """

    if F.shape[0] <= n_points:
        return F

    span = F.max(axis=0) - F.min(axis=0)
    Z = (F - F.min(axis=0)) / np.where(span > 0, span, 1.0)

    selected = list(dict.fromkeys(np.argmin(F, axis=0).tolist()))[:n_points]
    distance = np.min(
        np.linalg.norm(Z[:, None, :] - Z[None, selected, :], axis=2), axis=1
    )
    while len(selected) < n_points:
        idx = int(np.argmax(distance))
        selected.append(idx)
        distance = np.minimum(distance, np.linalg.norm(Z - Z[idx], axis=1))

    return F[np.sort(selected)]


def _filtered_front(image, n_points: int) -> np.ndarray:
    """Front of a problem whose sampled image contains dominated parts.

    The image is oversampled, filtered and reduced back to `n_points` spread
    points, so that sampling gaps do not leave dominated points in the front.

    Args:
        image: Callable mapping a number of samples to raw front points (P, M).
        n_points: Maximum number of points.

    Returns:
        (np.ndarray): Non-dominated points of shape (P, M), P <= n_points.

    """

    front = _non_dominated(image(_OVERSAMPLING * n_points))

    return _farthest_points(front, n_points)
