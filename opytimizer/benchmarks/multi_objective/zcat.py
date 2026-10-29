"""ZCAT family of scalable multi- and many-objective benchmark problems.

Each problem is f_i(x) = i^2 (F_i(y_I) + Z(J_i(y_II))), where y is x normalized
to [0, 1], F_i defines the shape of the front (from the first M - 1, or 1,
variables y_I) and Z the distance to it (from the remaining ones y_II). Problem
features are configurable: level of the distance landscape, bias, imbalance
and a complicated (curved) Pareto set.

References:
    S. Zapotecas-Martínez, C. A. Coello Coello, H. E. Aguirre, and K. Tanaka,
    "Challenging test problems for multi- and many-objective optimization,"
    Swarm and Evolutionary Computation, vol. 81, art. 101350, 2023.

"""

import math
from typing import Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.benchmarks._base import _MultiObjectiveBenchmark
from opytimizer.benchmarks._utils import _filtered_front, _product_shape, _unit_grid
from opytimizer.core.environment import Environment

# g function that defines the complicated Pareto set of each problem
_COMPLICATED_G = {
    1: 4, 2: 5, 3: 2, 4: 7, 5: 9, 6: 4, 7: 5, 8: 2, 9: 7, 10: 9,
    11: 3, 12: 10, 13: 1, 14: 6, 15: 8, 16: 10, 17: 1, 18: 8, 19: 6, 20: 3,
}  # fmt: skip

# Problems whose Pareto set is one-dimensional
_ONE_DIMENSIONAL_PS = (14, 15, 16)

# Ranges of y_1 where the front of ZCAT19 / ZCAT20 is degenerate
_DEGENERATE_RANGES = {19: ((0.0, 0.2), (0.4, 0.6)), 20: ((0.1, 0.4), (0.6, 0.9))}

_E8 = math.exp(8.0) - 1.0


def _in_range(y, low, high, eps):
    """Element-wise low <= y <= high, with a tolerance of `eps`."""
    return (y >= low - eps) & (y <= high + eps)


def _sum_shape(p, q, xp):
    """Averaged counterpart of `_product_shape`.

    f_1 = mean(p_1..p_{M-1}) and f_j = (p_1 + ... + p_{M-j} + q_{M-j+1}) / (M-j+1).
    """
    n, m1 = p.shape
    zeros = xp.zeros((n, 1), dtype=p.dtype)

    prefix = xp.concatenate((zeros, xp.cumsum(p, axis=1)), axis=1)
    closing = xp.concatenate((zeros, q[:, ::-1]), axis=1)

    counts = xp.arange(m1 + 1, 0, -1, dtype=p.dtype)
    counts[0] = m1

    return (prefix[:, ::-1] + closing) / counts


def _deform(y1, k, xp):
    """Cosine deformation of the last objective (ZCAT11-13, 15 and 16)."""
    return (
        xp.cos((2.0 * k - 1.0) * y1 * math.pi) + 2.0 * y1 + 4.0 * k * (1.0 - y1) - 1.0
    ) / (4.0 * k)


def _with_last(Y, last, xp):
    """Concatenates the first M - 1 objectives with the last one."""
    return xp.concatenate((Y, last[:, None]), axis=1)


def _shape(problem_id, y, m, eps, xp):
    """Shape functions F_1..F_M of each problem, of shape (N, M)."""
    y1 = y[:, 0]
    Y = y[:, : m - 1]
    half_pi = 0.5 * math.pi

    if problem_id == 1:
        F = _product_shape(xp.sin(half_pi * Y), xp.cos(half_pi * Y), xp)
        F[:, -1] = 1.0 - xp.sin(half_pi * y1)
        return xp.clip(F, 0.0, 1.0)
    if problem_id == 2:
        return _product_shape(1.0 - xp.cos(half_pi * Y), 1.0 - xp.sin(half_pi * Y), xp)
    if problem_id == 3:
        return _sum_shape(Y, 1.0 - Y, xp)
    if problem_id == 4:
        return _with_last(Y, 1.0 - xp.mean(Y, axis=1), xp)
    if problem_id == 5:
        return _with_last(Y, (xp.exp(8.0 * xp.mean(1.0 - Y, axis=1)) - 1.0) / _E8, xp)
    if problem_id == 6:
        k, r = 40.0, 0.05
        mu = xp.mean(Y, axis=1)
        num = 1.0 / (1.0 + xp.exp(2.0 * k * mu - k)) - r * mu - 1.0 / (1.0 + math.exp(k)) + r
        den = 1.0 / (1.0 + math.exp(-k)) - 1.0 / (1.0 + math.exp(k)) + r
        return _with_last(Y, num / den, xp)
    if problem_id == 7:
        last = xp.sum((0.5 - Y) ** 5, axis=1) / (2.0 * (m - 1) * 0.5**5) + 0.5
        return _with_last(Y, last, xp)
    if problem_id == 8:
        return 1.0 - _product_shape(1.0 - xp.sin(half_pi * Y), 1.0 - xp.cos(half_pi * Y), xp)
    if problem_id == 9:
        return _sum_shape(xp.sin(half_pi * Y), xp.cos(half_pi * Y), xp)
    if problem_id == 10:
        r = 0.02
        last = (1.0 / r - 1.0 / (xp.mean(1.0 - Y, axis=1) + r)) / (1.0 / r - 1.0 / (1.0 + r))
        return _with_last(Y, last, xp)
    if problem_id == 11:
        F = _sum_shape(Y, 1.0 - Y, xp)
        F[:, -1] = _deform(y1, 4.0, xp)
        return F
    if problem_id == 12:
        F = 1.0 - _product_shape(1.0 - Y, Y, xp)
        F[:, -1] = _deform(y1, 3.0, xp)
        return F
    if problem_id == 13:
        F = 1.0 - _sum_shape(xp.sin(half_pi * Y), xp.cos(half_pi * Y), xp)
        F[:, -1] = 1.0 - _deform(y1, 3.0, xp)
        return F
    if problem_id in (14, 16):
        # Powers of sin(y_1 pi / 2), plus an oscillating (M-1)-th objective
        s = xp.sin(half_pi * y1)
        base, freq = (2.0, 3.0) if problem_id == 14 else (1.0, 5.0)
        columns = [s**base]
        columns += [s ** (base + (j - 1.0) / (m - 2.0)) for j in range(2, m - 1)]
        if m > 2:
            columns.append(0.5 * (1.0 + xp.sin(freq * math.pi * y1 - half_pi)))
        columns.append(xp.cos(half_pi * y1) if problem_id == 14 else _deform(y1, 5.0, xp))
        return xp.stack(columns, axis=1)
    if problem_id == 15:
        columns = [y1 ** (1.0 + (j - 1.0) / (4.0 * m)) for j in range(1, m)]
        columns.append(_deform(y1, 3.0, xp))
        return xp.stack(columns, axis=1)
    if problem_id == 17:
        wedge = xp.all((Y >= 0.0) & (Y <= 0.5), axis=1)
        last = xp.where(
            wedge,
            (xp.exp(8.0 * (1.0 - y1)) - 1.0) / _E8,
            (xp.exp(8.0 * xp.mean(1.0 - Y, axis=1)) - 1.0) / _E8,
        )
        return _with_last(xp.where(wedge[:, None], y1[:, None], Y), last, xp)
    if problem_id == 18:
        wedge = xp.all(_in_range(Y, 0.0, 0.4, eps), axis=1) | xp.all(
            _in_range(Y, 0.6, 1.0, eps), axis=1
        )
        last = xp.where(
            wedge,
            ((0.5 - y1) ** 5 + 0.5**5) / (2.0 * 0.5**5),
            xp.sum((0.5 - Y) ** 5, axis=1) / (2.0 * (m - 1) * 0.5**5) + 0.5,
        )
        return _with_last(xp.where(wedge[:, None], y1[:, None], Y), last, xp)
    if problem_id == 19:
        a = 5.0
        flag = _degenerate_mask(19, y1, eps, xp)
        mu = xp.where(flag, y1, xp.mean(Y, axis=1))
        last = 1.0 - mu - xp.cos(2.0 * a * math.pi * mu + half_pi) / (2.0 * a * math.pi)
        F = _with_last(xp.where(flag[:, None], y1[:, None], Y), last, xp)
        return xp.clip(F, 0.0, 1.0)
    if problem_id == 20:
        flag = _degenerate_mask(20, y1, eps, xp)
        last = xp.where(
            flag,
            ((0.5 - y1) ** 5 + 0.5**5) / (2.0 * 0.5**5),
            xp.sum((0.5 - Y) ** 5, axis=1) / (2.0 * (m - 1) * 0.5**5) + 0.5,
        )
        return _with_last(xp.where(flag[:, None], y1[:, None], Y), last, xp)

    raise e.ValueError(f"Unsupported ZCAT problem: {problem_id}")


def _degenerate_mask(problem_id, y1, eps, xp):
    """Rows of ZCAT19 / ZCAT20 that lie in a degenerate region of the front."""
    (a, b), (c, d) = _DEGENERATE_RANGES[problem_id]

    return _in_range(y1, a, b, eps) | _in_range(y1, c, d, eps)


def _g(g_id, Y, n, xp):
    """Pareto set functions g_0..g_10, of shape (N, n - d), for Y = y_1..y_d."""
    N, d = Y.shape
    w = n - d
    theta = 2.0 * math.pi * xp.arange(w, dtype=Y.dtype) / w

    if g_id == 0:
        return xp.full((N, w), 0.2210, dtype=Y.dtype)

    # (N, d, 1) against (1, 1, n - d): one term per (y_i, theta_j)
    Yi = Y[:, :, None]
    mu = xp.mean(Y, axis=1, keepdims=True)

    if g_id == 1:
        return xp.sum(xp.sin(1.5 * math.pi * Yi + theta), axis=1) / (2.0 * d) + 0.5
    if g_id == 2:
        return xp.sum(Yi**2 * xp.sin(4.5 * math.pi * Yi + theta), axis=1) / (2.0 * d) + 0.5
    if g_id == 3:
        return xp.sum(xp.cos(math.pi * Yi + theta) ** 2, axis=1) / d
    if g_id == 4:
        return 0.5 * mu * xp.cos(4.0 * math.pi * mu + theta) + 0.5
    if g_id == 5:
        return xp.sum(xp.sin(2.0 * math.pi * Yi - 1.0 + theta) ** 3, axis=1) / (2.0 * d) + 0.5
    if g_id == 6:
        s1 = xp.mean(Y**2, axis=1, keepdims=True)
        s2 = xp.mean(xp.cos(11.0 * math.pi * Yi + theta) ** 3, axis=1)
        num = -10.0 * xp.exp(-0.4 * xp.sqrt(s1)) - xp.exp(s2) + 10.0 + math.e
        den = -10.0 * math.exp(-0.4) - math.exp(-1.0) + 10.0 + math.e
        return num / den
    if g_id == 7:
        num = mu + xp.exp(xp.sin(7.0 * math.pi * mu - 0.5 * math.pi + theta)) - math.exp(-1.0)
        return num / (1.0 + math.e - math.exp(-1.0))
    if g_id == 8:
        return xp.sum(xp.abs(xp.sin(2.5 * math.pi * (Yi - 0.5) + theta)), axis=1) / d
    if g_id == 9:
        total = xp.sum(xp.abs(xp.sin(2.5 * math.pi * Yi - 0.5 * math.pi + theta)), axis=1)
        return 0.5 * mu - total / (2.0 * d) + 0.5
    if g_id == 10:
        total = xp.sum(xp.sin((4.0 * Yi - 2.0) * math.pi + theta), axis=1)
        return total**3 / (2.0 * d**3) + 0.5

    raise e.ValueError(f"Unsupported g function: {g_id}")


def _z(level, J, xp):
    """Distance functions Z_1..Z_6 over the variables J of shape (N, |J|), Z(0) = 0."""
    size = J.shape[1]
    A = xp.abs(J)
    cosine = xp.cos(9.0 * math.pi * J)

    if level == 1:
        return 10.0 * xp.sum(J**2, axis=1) / size
    if level == 2:
        return 10.0 * xp.max(A, axis=1)
    if level == 3:
        return 10.0 * xp.sum((J**2 - cosine + 1.0) / 3.0, axis=1) / size
    if level == 4:
        num = (
            xp.exp(xp.sqrt(xp.max(A, axis=1)))
            - xp.exp(xp.mean(0.5 * (cosine + 1.0), axis=1))
            - 1.0
            + math.e
        )
        return 10.0 / (2.0 * math.e - 2.0) * num
    if level == 5:
        return -0.7 * _z(3, J, xp) + 10.0 * xp.sum(A**0.002, axis=1) / size
    if level == 6:
        return -0.7 * _z(4, J, xp) + 10.0 * xp.mean(A, axis=1) ** 0.002

    raise e.ValueError(f"Unsupported Z level: {level}")


class _ZCAT(_MultiObjectiveBenchmark):
    """Base class of the ZCAT problems."""

    _problem_id = 0

    def __init__(
        self,
        n_variables: int = 30,
        n_objectives: int = 2,
        level: int = 1,
        bias: bool = False,
        imbalance: bool = False,
        complicated_pareto_set: bool = False,
        env: Optional[Environment] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables (at least n_objectives - 1,
                or 1 for ZCAT14-16).
            n_objectives: Number of objectives (at least 2).
            level: Difficulty of the distance landscape, from 1 (unimodal) to 6.
            bias: Whether a bias towards the Pareto set is introduced.
            imbalance: Whether odd and even objectives use different landscapes.
            complicated_pareto_set: Whether the Pareto set is curved (g_1..g_10)
                instead of a constant (g_0).
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        if not isinstance(n_objectives, (int, np.integer)) or n_objectives < 2:
            raise e.ValueError("ZCAT problems require `n_objectives` >= 2")
        if level not in range(1, 7):
            raise e.ValueError("`level` should be an integer in [1, 6]")

        m = int(n_objectives)
        min_n = 1 if self._problem_id in _ONE_DIMENSIONAL_PS else m - 1
        if not isinstance(n_variables, (int, np.integer)) or n_variables < max(1, min_n):
            raise e.ValueError(f"{self.name} requires `n_variables` >= {max(1, min_n)}")

        self.level = int(level)
        self.bias = bool(bias)
        self.imbalance = bool(imbalance)
        self.complicated_pareto_set = bool(complicated_pareto_set)

        # x_i in [-i / 2, i / 2]
        half = 0.5 * np.arange(1, n_variables + 1, dtype=float)
        super().__init__(int(n_variables), m, -half, half, env)

    @property
    def _ps_dimension(self) -> int:
        """Dimension d of the Pareto set (ZCAT19 / ZCAT20 also use d = 1 in places)."""
        return 1 if self._problem_id in _ONE_DIMENSIONAL_PS else self.n_objectives - 1

    def _beta(self, y, d, xp):
        """Distance term Z(J_i) of every objective for a Pareto set of dimension d."""
        n, m = self.n_variables, self.n_objectives
        if d >= n:
            return xp.zeros((y.shape[0], m), dtype=y.dtype)

        g_id = _COMPLICATED_G[self._problem_id] if self.complicated_pareto_set else 0
        z = y[:, d:] - _g(g_id, y[:, :d], n, xp)
        w = xp.abs(z) ** 0.05 if self.bias else z

        columns = []
        for i in range(1, m + 1):
            # J_i: every M-th variable of y_II starting at the i-th one
            J = w[:, i - 1 :: m] if i - 1 < w.shape[1] else w[:, :1]
            level = (4 if i % 2 == 0 else 1) if self.imbalance else self.level
            columns.append(_z(level, J, xp))

        return xp.stack(columns, axis=1)

    def _evaluate(self, X, xp):
        y = xp.clip(self._normalize(X), 0.0, 1.0)
        eps = float(xp.finfo(y.dtype).eps)

        alpha = _shape(self._problem_id, y, self.n_objectives, eps, xp)
        beta = self._beta(y, self._ps_dimension, xp)

        if self._problem_id in _DEGENERATE_RANGES:
            # Inside the degenerate regions the Pareto set is one-dimensional
            flag = _degenerate_mask(self._problem_id, y[:, 0], eps, xp)
            beta = xp.where(flag[:, None], self._beta(y, 1, xp), beta)

        squares = xp.arange(1, self.n_objectives + 1, dtype=y.dtype) ** 2

        return squares * (alpha + beta)

    def _front_image(self, n_points):
        """Image i^2 F_i of the Pareto set (Z = 0), including its dominated parts."""
        m = self.n_objectives
        y = _unit_grid(self._ps_dimension, n_points)
        if y.shape[1] < m - 1:
            y = np.concatenate((y, np.zeros((y.shape[0], m - 1 - y.shape[1]))), axis=1)

        F = _shape(self._problem_id, y, m, np.finfo(float).eps, np)
        squares = np.arange(1, m + 1, dtype=float) ** 2

        return squares * F

    def _pareto_front(self, n_points):
        return _filtered_front(self._front_image, n_points)


class ZCAT1(_ZCAT):
    """ZCAT1: concave front."""

    _problem_id = 1


class ZCAT2(_ZCAT):
    """ZCAT2: convex front."""

    _problem_id = 2


class ZCAT3(_ZCAT):
    """ZCAT3: linear front."""

    _problem_id = 3


class ZCAT4(_ZCAT):
    """ZCAT4: linear front (direct assignment of the position variables)."""

    _problem_id = 4


class ZCAT5(_ZCAT):
    """ZCAT5: convex front with an exponential last objective."""

    _problem_id = 5


class ZCAT6(_ZCAT):
    """ZCAT6: front with a logistic last objective."""

    _problem_id = 6


class ZCAT7(_ZCAT):
    """ZCAT7: front with a quintic last objective."""

    _problem_id = 7


class ZCAT8(_ZCAT):
    """ZCAT8: convex front (inverted concave shape)."""

    _problem_id = 8


class ZCAT9(_ZCAT):
    """ZCAT9: concave front (averaged shape)."""

    _problem_id = 9


class ZCAT10(_ZCAT):
    """ZCAT10: concave front with a reciprocal last objective."""

    _problem_id = 10


class ZCAT11(_ZCAT):
    """ZCAT11: disconnected linear front."""

    _problem_id = 11


class ZCAT12(_ZCAT):
    """ZCAT12: disconnected front."""

    _problem_id = 12


class ZCAT13(_ZCAT):
    """ZCAT13: disconnected front (averaged shape)."""

    _problem_id = 13


class ZCAT14(_ZCAT):
    """ZCAT14: degenerate front with a one-dimensional Pareto set."""

    _problem_id = 14


class ZCAT15(_ZCAT):
    """ZCAT15: degenerate front with a one-dimensional Pareto set."""

    _problem_id = 15


class ZCAT16(_ZCAT):
    """ZCAT16: degenerate disconnected front with a one-dimensional Pareto set."""

    _problem_id = 16


class ZCAT17(_ZCAT):
    """ZCAT17: front with a degenerate wedge region."""

    _problem_id = 17


class ZCAT18(_ZCAT):
    """ZCAT18: front with two degenerate wedge regions."""

    _problem_id = 18


class ZCAT19(_ZCAT):
    """ZCAT19: hybrid front, degenerate for some values of y_1."""

    _problem_id = 19


class ZCAT20(_ZCAT):
    """ZCAT20: hybrid front with two degenerate regions."""

    _problem_id = 20
