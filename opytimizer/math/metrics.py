import multiprocessing
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Union

import numpy as np

# Upper bound on the number of elements of each temporary of the blocked
# pairwise computations (distances and dominance checks)
_DOMINANCE_BLOCK_ELEMENTS = 1 << 24


def _to_host(value: Any) -> Any:
    """Host (NumPy) copy of `value` when it lives on the GPU (CuPy), else `value`.

    Metrics are computed with NumPy, whatever the environment of the task: the
    fronts are small, so a single device -> host copy per evaluation is cheap.
    CuPy is detected through `__cuda_array_interface__`, so it is never imported.
    """

    if hasattr(value, "__cuda_array_interface__"):
        return value.get()

    if isinstance(value, tuple):
        return tuple(_to_host(item) for item in value)

    return value


def _as_host_array(value: Any) -> np.ndarray:
    """2-D float NumPy array of `value` (e.g., a reference front or point set)."""
    return np.atleast_2d(np.asarray(_to_host(value), dtype=float))


def _min_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of `A` to its nearest row of `B`."""
    out = np.empty(A.shape[0])
    if A.shape[0] == 0 or B.shape[0] == 0:
        out.fill(np.inf)
        return out

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, so each block is a matrix product
    # instead of an (block, |B|, M) temporary. The expansion only selects the
    # nearest row; the distance itself is recomputed directly, so coincident
    # points get exactly zero instead of round-off noise
    b_norms = np.sum(B**2, axis=1)
    block = max(1, _DOMINANCE_BLOCK_ELEMENTS // B.shape[0])
    for start in range(0, A.shape[0], block):
        a = A[start : start + block]
        squared = b_norms[None, :] - 2.0 * (a @ B.T)
        nearest = B[np.argmin(squared, axis=1)]
        out[start : start + block] = np.linalg.norm(a - nearest, axis=1)

    return out


class _BaseMetric(ABC):
    @property
    def pareto_front(self) -> Any:
        return self._pareto_front

    @pareto_front.setter
    def pareto_front(self, value) -> None:
        if value is None:
            return

        # wrapper object
        if hasattr(value, "pareto_front"):
            value = value.pareto_front

        # Fronts of tensorized GPU spaces are CuPy arrays (or tuples of them)
        value = _to_host(value)

        # Tuple (positions, fits)

        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[1], np.ndarray)
            and value[1].ndim == 2
        ):
            points = value[1]

        # numpy array
        elif isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
            points = value

        # 1 element tuple, where fit is a scalar or 1d numpy array
        elif (
            isinstance(value, tuple)
            and len(value) == 2
            and not isinstance(value[1], (tuple, list))
        ):
            points = np.atleast_2d(value[1])

        # Elements collections
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            first = value[0]
            # Has .fit atributes
            if hasattr(first, "fit"):
                points = np.array([_to_host(agent.fit) for agent in value])

            #  Tuple list [(pos, fit), (pos, fit), ...]
            elif isinstance(first, (tuple, list, np.ndarray)) and len(first) == 2:
                points = np.array([_to_host(item[1]) for item in value])

            # Fitness array (e.g, [[f1, f2], [f1, f2]])
            else:
                points = np.array([_to_host(item) for item in value])

        # Fallback
        else:
            points = np.atleast_2d(value)

        # Ensure (N_points, N_objs) shape
        self._pareto_front = np.atleast_2d(points).astype(float)

    @abstractmethod
    def __call__(self, pareto_front):
        pass

    @property
    def name(self):
        return type(self).__name__


class IGD(_BaseMetric):
    def __init__(self, pareto_optimal):
        self.pareto_optimal = _as_host_array(pareto_optimal)
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(
                self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals
            )
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0 or self.min_vals is None:
            return 0.0
        normalized_front = (self.pareto_front - self.min_vals) / self.denom
        # Mean distance from each reference point to the obtained front
        distances = _min_distances(self.normalized_optimal, normalized_front)
        return float(np.mean(distances))


class GD(_BaseMetric):
    def __init__(self, pareto_optimal):
        self.pareto_optimal = _as_host_array(pareto_optimal)
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(
                self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals
            )
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0 or self.min_vals is None:
            return 0.0
        normalized_front = (self.pareto_front - self.min_vals) / self.denom
        # Mean distance from each obtained solution to the reference front
        distances = _min_distances(normalized_front, self.normalized_optimal)
        return float(np.mean(distances))


class Spread(_BaseMetric):
    def __init__(self, pareto_optimal):
        self.pareto_optimal = _as_host_array(pareto_optimal)
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(
                self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals
            )
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.shape[0] < 2 or self.min_vals is None:
            return 0.0
        pf = (self.pareto_front - self.min_vals) / self.denom
        pf = pf[np.lexsort(pf.T[::-1])]
        df = np.linalg.norm(pf[1:] - pf[:-1], axis=1)
        d_mean = np.mean(df)

        # Distances between the boundary solutions of the obtained front and the
        # extreme solutions of the reference front (Deb et al., 2002)
        extremes = self.normalized_optimal[np.lexsort(self.normalized_optimal.T[::-1])]
        d_f = np.linalg.norm(pf[0] - extremes[0])
        d_l = np.linalg.norm(pf[-1] - extremes[-1])

        denominator = d_f + d_l + len(df) * d_mean
        if denominator == 0:
            return 0.0

        return float((d_f + d_l + np.sum(np.abs(df - d_mean))) / denominator)


class ErrorRatio(_BaseMetric):
    def __init__(self, pareto_optimal, tol=1e-6):
        self.pareto_optimal = _as_host_array(pareto_optimal)
        self.tol = tol
        if self.pareto_optimal.size == 0:
            self.min_vals = None
            self.denom = None
            self.normalized_optimal = None
        else:
            self.min_vals = np.min(self.pareto_optimal, axis=0)
            self.max_vals = np.max(self.pareto_optimal, axis=0)
            self.denom = np.where(
                self.max_vals - self.min_vals == 0, 1.0, self.max_vals - self.min_vals
            )
            self.normalized_optimal = (self.pareto_optimal - self.min_vals) / self.denom

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0 or self.min_vals is None:
            return 0.0
        pf = (self.pareto_front - self.min_vals) / self.denom
        # Fraction of solutions farther than `tol` from the reference front
        errors = _min_distances(pf, self.normalized_optimal) > self.tol
        return float(np.mean(errors))


class R2(_BaseMetric):
    def __init__(self, weight_vectors, ideal_point=None, nadir_point=None):
        self.weight_vectors = _as_host_array(weight_vectors)
        norms = np.linalg.norm(self.weight_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.weight_vectors = self.weight_vectors / norms
        self.ideal_point = (
            None if ideal_point is None else _as_host_array(ideal_point)[0]
        )
        self.nadir_point = (
            None if nadir_point is None else _as_host_array(nadir_point)[0]
        )

    def __call__(self, pareto_front):

        self.pareto_front = pareto_front
        if self.pareto_front.size == 0:
            return 0.0
        current_ideal = (
            self.ideal_point
            if self.ideal_point is not None
            else np.min(self.pareto_front, axis=0)
        )
        current_nadir = (
            self.nadir_point
            if self.nadir_point is not None
            else np.max(self.pareto_front, axis=0)
        )
        denom = np.where(
            current_nadir - current_ideal == 0, 1.0, current_nadir - current_ideal
        )
        normalized_pf = (self.pareto_front - current_ideal) / denom
        # (W, N) Tchebycheff values of every solution for every weight vector
        tchebycheff = np.max(
            self.weight_vectors[:, None, :] * normalized_pf[None, :, :], axis=2
        )
        return float(np.mean(np.min(tchebycheff, axis=1)))


class MaximumSpread(_BaseMetric):
    """Maximum spread: diagonal of the bounding box of the obtained front.

    MS = sqrt(sum_m (max f_m - min f_m)^2), as defined by Zitzler, Deb and
    Thiele (2000).
    """

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.shape[0] < 2:
            return 0.0
        extent = np.max(self.pareto_front, axis=0) - np.min(self.pareto_front, axis=0)
        return float(np.sqrt(np.sum(extent**2)))


def _dominated_by(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Mask of the rows of `C` dominated by at least one row of `A`."""
    if A.shape[0] == 0:
        return np.zeros(C.shape[0], dtype=bool)

    no_worse = np.all(A[None, :, :] <= C[:, None, :], axis=2)
    better = np.any(A[None, :, :] < C[:, None, :], axis=2)

    return np.any(no_worse & better, axis=1)


def _non_dominated(F: np.ndarray) -> np.ndarray:
    """Removes duplicated and dominated rows of `F` (minimization).

    After a lexicographic sort, a dominating row always precedes the rows it
    dominates, so each row is only compared with the non-dominated rows found
    before it.
    """

    F = np.unique(F, axis=0)
    n, m = F.shape
    if n < 2:
        return F

    if m == 2:
        # Sorted by f_1: a row survives iff it improves the best f_2 seen so far
        best = np.minimum.accumulate(np.concatenate(([np.inf], F[:-1, 1])))
        return F[F[:, 1] < best]

    archive = F[:0]
    start = 0
    while start < n:
        # Block size keeps the (block, archive, M) temporaries bounded
        budget = _DOMINANCE_BLOCK_ELEMENTS // max(1, archive.shape[0] * m)
        size = max(1, min(1024, budget))
        C = F[start : start + size]
        dominated = _dominated_by(archive, C) | _dominated_by(C, C)
        archive = np.concatenate((archive, C[~dominated]), axis=0)
        start += size

    return archive


def _hv_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume by a sweep over the first objective."""
    F = F[np.lexsort((F[:, 1], F[:, 0]))]

    # Staircase: a point survives iff it improves the best f_2 seen so far
    best = np.minimum.accumulate(np.concatenate(([np.inf], F[:-1, 1])))
    F = F[F[:, 1] < best]

    widths = np.append(F[1:, 0], ref[0]) - F[:, 0]

    return float(np.sum(widths * (ref[1] - F[:, 1])))


def _hv_3d(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact 3-D hypervolume by slicing along the third objective.

    Between two consecutive values of f_3, the dominated region is a prism whose
    base is the 2-D hypervolume of the points already swept.
    """

    F = F[np.argsort(F[:, 2], kind="stable")]
    heights = np.append(F[1:, 2], ref[2]) - F[:, 2]

    volume = 0.0
    for i in np.flatnonzero(heights > 0):
        volume += heights[i] * _hv_2d(F[: i + 1, :2], ref[:2])

    return float(volume)


def _hv(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume of points strictly better than `ref` in every objective.

    Up to three objectives, dimension-sweep algorithms are used. Beyond that, the
    WFG algorithm is applied with slicing on the last objective: once the points
    are sorted by decreasing f_M, the limit set of point k is constant in f_M,
    so its exclusive hypervolume reduces to an (M - 1)-dimensional problem.

    References:
        L. While, L. Bradstreet, and L. Barone, "A fast way of calculating exact
        hypervolumes," IEEE Trans. Evol. Comput., vol. 16, no. 1, pp. 86-95, 2012.
    """

    n, m = F.shape
    if n == 0:
        return 0.0
    if m == 1:
        return float(ref[0] - F[:, 0].min())
    if m == 2:
        return _hv_2d(F, ref)
    if n == 1:
        return float(np.prod(ref - F[0]))
    if m == 3:
        return _hv_3d(F, ref)

    F = F[np.argsort(-F[:, -1], kind="stable")]

    volume = 0.0
    for k in range(n):
        head, last = F[k, :-1], F[k, -1]
        # Points after k are no worse in f_M, so they only matter through the
        # first M - 1 objectives, bounded by point k
        limit = _non_dominated(np.maximum(F[k + 1 :, :-1], head))
        exclusive = np.prod(ref[:-1] - head) - _hv(limit, ref[:-1])
        volume += (ref[-1] - last) * exclusive

    return float(volume)


class HV(_BaseMetric):
    """Exact hypervolume, for any number of objectives (minimization).

    The hypervolume is the volume of the objective space dominated by the front
    and bounded by `reference_point`. Points that are not strictly better than
    the reference point in every objective do not contribute to it.
    """

    def __init__(self, reference_point=[1.01, 1.01]):
        self.reference_point = _as_host_array(reference_point)[0]

    def __call__(self, pareto_front):

        self.pareto_front = pareto_front

        if self.pareto_front.ndim > 2:
            self.pareto_front = self.pareto_front.squeeze()

        self.pareto_front = np.atleast_2d(self.pareto_front)

        if self.pareto_front.size == 0:
            return 0.0

        if self.pareto_front.shape[1] != self.reference_point.shape[0]:
            raise ValueError(
                f"The front has {self.pareto_front.shape[1]} objectives, but the "
                f"reference point has {self.reference_point.shape[0]}."
            )

        # Only points strictly inside the reference box contribute
        inside = np.all(self.pareto_front < self.reference_point, axis=1)
        front = self.pareto_front[inside]

        # Sweep algorithms (M <= 3) skip dominated points by themselves
        if front.shape[1] > 3:
            front = _non_dominated(front)

        return _hv(front, self.reference_point)


def _check_dominance_worker(samples_chunk, pareto_front):

    is_dominated = np.zeros(len(samples_chunk), dtype=bool)

    for pareto_point in pareto_front:

        remaining = ~is_dominated
        if not np.any(remaining):
            break

        diff = pareto_point - samples_chunk[remaining]
        domination_mask = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
        is_dominated[remaining] = domination_mask

    return np.sum(is_dominated)


class MonteCarloHV(_BaseMetric):
    """Hypervolume estimated by Monte Carlo sampling (suited for many objectives).

    Samples are drawn uniformly from the box [lower_bound, upper_bound], and the
    hypervolume is the fraction of samples dominated by the front times the
    volume of the box. `upper_bound` plays the role of the reference point and
    `lower_bound` should be no worse than the ideal point of the front.
    """

    def __init__(
        self,
        n_samples: int = 10**6,
        lower_bound: Union[float, List[float]] = 0.0,
        upper_bound: Union[float, List[float]] = 1.0,
        parallel: bool = False,
    ):
        """Initialization method.

        Args:
            n_samples: Number of Monte Carlo samples.
            lower_bound: Lower corner of the sampling box (scalar or per objective).
            upper_bound: Upper corner of the sampling box, i.e., the reference point.
            parallel: Whether the dominance checks are split among threads.
            diff: Deprecated and ignored. It used to enlarge the measured volume
                without enlarging the sampled box, which biased the estimate.

        """

        

        self.lower_bound = np.array(_to_host(lower_bound), dtype=float)
        self.upper_bound = np.array(_to_host(upper_bound), dtype=float)
        self.n_samples = int(n_samples)
    
        # Scalar bounds are expanded to the number of objectives on the first call
        self.samples = None
        if self.lower_bound.ndim > 0 or self.upper_bound.ndim > 0:
            self._sample(np.broadcast(self.lower_bound, self.upper_bound).size)

        self._exec = self._process_parallel if parallel else self._process_serial

    def _box(self, n_objectives):
        """Lower and upper corners of the sampling box, one value per objective."""
        low = np.broadcast_to(self.lower_bound, (n_objectives,))
        high = np.broadcast_to(self.upper_bound, (n_objectives,))

        return low, high

    def _sample(self, n_objectives):
        """Draws the samples of the box for `n_objectives` objectives."""
        low, high = self._box(n_objectives)
        self.n_vars = n_objectives
        self.samples = np.random.uniform(
            low=low, high=high, size=(self.n_samples, n_objectives)
        )

    def _calculate_final_hv(self, count):
        low, high = self._box(self.n_vars)
        return float(count / self.n_samples * np.prod(high - low))

    def _process_serial(self, pareto_front):
        count = _check_dominance_worker(self.samples, pareto_front)
        return self._calculate_final_hv(count)

    def _process_parallel(self, pareto_front):
        n_cores = multiprocessing.cpu_count()
        chunks = np.array_split(self.samples, n_cores)

        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(_check_dominance_worker, chunk, pareto_front)
                for chunk in chunks
            ]
            total_count = sum(f.result() for f in futures)

        return self._calculate_final_hv(total_count)

    def __call__(self, pareto_front):
        self.pareto_front = pareto_front
        if self.pareto_front.size == 0:
            return 0.0

        n_objectives = self.pareto_front.shape[1]
        if self.samples is None or self.samples.shape[1] != n_objectives:
            self._sample(n_objectives)

        # Dominated points add no volume, so they are skipped
        front = _non_dominated(self.pareto_front)

        return self._exec(front)
