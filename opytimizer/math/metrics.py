from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    """Interface para todas as métricas multiobjetivo."""

    @abstractmethod
    def __call__(self, pareto_front, pareto_optimal=None, **kwargs):
        pass

    @property
    def name(self):
        return type(self).__name__


class IGDMetric(BaseMetric):
    def __call__(self, pareto_front, pareto_optimal, **kwargs):
        pareto_front = np.atleast_2d(pareto_front)
        pareto_optimal = np.atleast_2d(pareto_optimal)
        if pareto_front.size == 0 or pareto_optimal.size == 0:
            return 0.0
        min_vals = np.min(pareto_optimal, axis=0)
        max_vals = np.max(pareto_optimal, axis=0)
        denom = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        normalized_front = (pareto_front - min_vals) / denom
        normalized_optimal = (pareto_optimal - min_vals) / denom
        distances = []
        for optimal in normalized_optimal:
            d = np.linalg.norm(normalized_front - optimal, axis=1)
            distances.append(np.min(d))
        return np.mean(distances)


class GDMetric(BaseMetric):
    def __call__(self, pareto_front, pareto_optimal, **kwargs):
        pareto_front = np.atleast_2d(pareto_front)
        pareto_optimal = np.atleast_2d(pareto_optimal)
        if pareto_front.size == 0 or pareto_optimal.size == 0:
            return 0.0
        min_vals = np.min(pareto_optimal, axis=0)
        max_vals = np.max(pareto_optimal, axis=0)
        denom = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        normalized_front = (pareto_front - min_vals) / denom
        normalized_optimal = (pareto_optimal - min_vals) / denom
        distances = []
        for solution in normalized_front:
            d = np.linalg.norm(normalized_optimal - solution, axis=1)
            distances.append(np.min(d))
        return np.mean(distances)


class SpreadMetric(BaseMetric):
    def __call__(self, pareto_front, pareto_optimal, **kwargs):
        pareto_front = np.atleast_2d(pareto_front)
        pareto_optimal = np.atleast_2d(pareto_optimal)
        if pareto_front.shape[0] < 2 or pareto_optimal.shape[0] == 0:
            return 0.0
        min_vals = np.min(pareto_optimal, axis=0)
        max_vals = np.max(pareto_optimal, axis=0)
        denom = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        pf = (pareto_front - min_vals) / denom
        po = (pareto_optimal - min_vals) / denom
        pf = pf[np.lexsort(np.rot90(pf))]
        df = np.linalg.norm(pf[1:] - pf[:-1], axis=1)
        d_mean = np.mean(df)
        d_f = np.min(np.linalg.norm(pf[0] - po, axis=1))
        d_l = np.min(np.linalg.norm(pf[-1] - po, axis=1))
        delta = (d_f + d_l + np.sum(np.abs(df - d_mean))) / (
            d_f + d_l + len(df) * d_mean
        )
        return delta


class ErrorRatioMetric(BaseMetric):
    def __call__(self, pareto_front, pareto_optimal, tol=1e-6, **kwargs):
        pareto_front = np.atleast_2d(pareto_front)
        pareto_optimal = np.atleast_2d(pareto_optimal)
        if pareto_front.size == 0 or pareto_optimal.size == 0:
            return 0.0
        min_vals = np.min(pareto_optimal, axis=0)
        max_vals = np.max(pareto_optimal, axis=0)
        denom = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        pf = (pareto_front - min_vals) / denom
        po = (pareto_optimal - min_vals) / denom
        errors = 0
        for solution in pf:
            distances = np.linalg.norm(po - solution, axis=1)
            if np.min(distances) > tol:
                errors += 1
        return errors / len(pf) if len(pf) > 0 else 0.0


class R2Metric(BaseMetric):
    def __call__(self, pareto_front, weight_vectors, **kwargs):
        pareto_front = np.atleast_2d(pareto_front)
        weight_vectors = np.atleast_2d(weight_vectors)
        if pareto_front.shape[0] == 0 or weight_vectors.shape[0] == 0:
            return 0.0
        ideal_point = np.min(pareto_front, axis=0)
        nadir_point = np.max(pareto_front, axis=0)
        denom = np.where(nadir_point - ideal_point == 0, 1.0, nadir_point - ideal_point)
        normalized_pf = (pareto_front - ideal_point) / denom
        r2_values = []
        for w in weight_vectors:
            w = w / np.linalg.norm(w)
            tchebycheff = np.max(normalized_pf * w, axis=1)
            r2_values.append(np.min(tchebycheff))
        return np.mean(r2_values)


class MaximumSpreadMetric(BaseMetric):
    def __call__(self, pareto_front, **kwargs):
        pareto_front = np.atleast_2d(pareto_front)
        if pareto_front.shape[0] < 2:
            return 0.0
        dists = np.linalg.norm(
            pareto_front[None, :, :] - pareto_front[:, None, :], axis=2
        )
        return np.max(dists)


class HypervolumeMetric(BaseMetric):
    def __call__(self, pareto_front, reference_point, **kwargs):
        pareto_front = np.atleast_2d(pareto_front)
        reference_point = np.asarray(reference_point)

        if pareto_front.shape[0] < 2 or pareto_front.shape[1] != 2:
            return 0.0

        # Sort by first objective (lower is better)
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]

        hv = 0.0
        prev_x = reference_point[0]

        for point in reversed(sorted_front):
            width = prev_x - point[0]
            height = reference_point[1] - point[1]
            if width > 0 and height > 0:
                hv += width * height
            prev_x = point[0]

        return hv
