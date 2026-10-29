"""WFG family of scalable multi- and many-objective benchmark problems.

Each problem maps z in prod[0, 2i] to f_m = x_M + S_m h_m(x_1, ..., x_{M-1}),
S_m = 2m, through a chain of transformations of the normalized variables. The
first k variables are position-related and the remaining l = n - k are
distance-related.

References:
    S. Huband, P. Hingston, L. Barone, and L. While, "A review of multiobjective
    test problems and a scalable test problem toolkit," IEEE Trans. Evol. Comput.,
    vol. 10, no. 5, pp. 477-506, 2006.

"""

import math
from typing import Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.benchmarks._base import _MultiObjectiveBenchmark
from opytimizer.benchmarks._utils import (
    _filtered_front,
    _product_shape,
    _unit_grid,
    _unit_sphere,
)
from opytimizer.core.environment import Environment

# Number of distance-related variables recommended by Huband et al. (2006)
_DEFAULT_L = 20


# Transformations (Huband et al., 2006, Table XI). Every output is clipped to
# [0, 1], which absorbs the round-off of low-precision (e.g., float32) runs.


def _s_linear(y, A, xp):
    """Linear shift: optimum moved to `A`."""
    return xp.clip(xp.abs(y - A) / xp.abs(xp.floor(A - y) + A), 0.0, 1.0)


def _s_decept(y, A, B, C, xp):
    """Deceptive shift: one global and two deceptive optima."""
    tmp1 = xp.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = xp.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)

    return xp.clip(1.0 + (xp.abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B), 0.0, 1.0)


def _s_multi(y, A, B, C, xp):
    """Multi-modal shift: `A` minima, hill size `B`, optimum at `C`."""
    tmp1 = xp.abs(y - C) / (2.0 * (xp.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * math.pi * (0.5 - tmp1)

    return xp.clip((1.0 + xp.cos(tmp2) + 4.0 * B * tmp1**2) / (B + 2.0), 0.0, 1.0)


def _b_poly(y, alpha, xp):
    """Polynomial bias."""
    return xp.clip(y**alpha, 0.0, 1.0)


def _b_flat(y, A, B, C, xp):
    """Flat region bias: values in [B, C] are mapped to `A`."""
    ret = (
        A
        + xp.minimum(0.0, xp.floor(y - B)) * A * (B - y) / B
        - xp.minimum(0.0, xp.floor(C - y)) * (1.0 - A) * (y - C) / (1.0 - C)
    )

    return xp.clip(ret, 0.0, 1.0)


def _b_param(y, u, A, B, C, xp):
    """Parameter-dependent bias, driven by the reduction `u` of other variables."""
    aux = A - (1.0 - 2.0 * u) * xp.abs(xp.floor(0.5 - u) + A)

    return xp.clip(y ** (B + (C - B) * aux), 0.0, 1.0)


def _r_sum(y, w, xp):
    """Weighted sum reduction over the last axis (`w=None` means uniform weights)."""
    if w is None:
        return xp.clip(xp.mean(y, axis=-1), 0.0, 1.0)

    return xp.clip(xp.sum(y * w, axis=-1) / xp.sum(w, axis=-1), 0.0, 1.0)


def _r_nonsep(y, A, xp):
    """Non-separable reduction over the last axis, with degree of non-separability `A`."""
    m = y.shape[-1]
    num = xp.sum(y, axis=-1)
    for k in range(A - 1):
        # |y_j - y_{(j + k + 1) mod m}| for every j
        num = num + xp.sum(xp.abs(y - xp.roll(y, -(k + 1), axis=-1)), axis=-1)

    half = math.ceil(A / 2.0)
    denom = m * half * (1.0 + 2.0 * A - 2.0 * half) / A

    return xp.clip(num / denom, 0.0, 1.0)


def _suffix_mean(y, xp):
    """mean(y_{i+1}, ..., y_n) for i = 1, ..., n - 1, of shape (N, n - 1)."""
    n = y.shape[1]
    tail = xp.cumsum(y[:, ::-1], axis=1)[:, ::-1]

    return tail[:, 1:] / xp.arange(n - 1, 0, -1, dtype=y.dtype)


def _prefix_mean(y, xp):
    """mean(y_1, ..., y_i) for i = 1, ..., n, of shape (N, n)."""
    n = y.shape[1]

    return xp.cumsum(y, axis=1) / xp.arange(1, n + 1, dtype=y.dtype)


# Shapes (Huband et al., 2006, Table IX)


def _concave(x, xp):
    """Concave shape (hypersphere)."""
    return _product_shape(xp.sin(0.5 * math.pi * x), xp.cos(0.5 * math.pi * x), xp)


def _convex(x, xp):
    """Convex shape."""
    return _product_shape(
        1.0 - xp.cos(0.5 * math.pi * x), 1.0 - xp.sin(0.5 * math.pi * x), xp
    )


def _linear(x, xp):
    """Linear shape (hyperplane)."""
    return _product_shape(x, 1.0 - x, xp)


def _mixed(x1, A, alpha, xp):
    """Mixed convex / concave shape with `A` segments."""
    aux = 2.0 * A * math.pi

    return xp.clip((1.0 - x1 - xp.cos(aux * x1 + 0.5 * math.pi) / aux) ** alpha, 0.0, 1.0)


def _disconnected(x1, A, alpha, beta, xp):
    """Disconnected shape with `A` regions."""
    return xp.clip(1.0 - x1**alpha * xp.cos(A * math.pi * x1**beta) ** 2, 0.0, 1.0)


class _WFG(_MultiObjectiveBenchmark):
    """Base class of the WFG problems."""

    def __init__(
        self,
        n_variables: Optional[int] = None,
        n_objectives: int = 3,
        k: Optional[int] = None,
        env: Optional[Environment] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables (k + l). Defaults to k + 20.
            n_objectives: Number of objectives (at least 2).
            k: Number of position-related variables, a multiple of
                n_objectives - 1. Defaults to 4 for two objectives and to
                2 (n_objectives - 1) otherwise.
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        if not isinstance(n_objectives, (int, np.integer)) or n_objectives < 2:
            raise e.ValueError("WFG problems require `n_objectives` >= 2")

        m = int(n_objectives)
        if k is None:
            k = 4 if m == 2 else 2 * (m - 1)
        if not isinstance(k, (int, np.integer)) or k <= 0 or k % (m - 1):
            raise e.ValueError("`k` should be a positive multiple of `n_objectives` - 1")

        n = k + _DEFAULT_L if n_variables is None else n_variables
        if not isinstance(n, (int, np.integer)) or n <= k:
            raise e.ValueError("WFG problems require `n_variables` > `k`")

        self.k = int(k)
        self.l = int(n) - self.k
        self._validate_l()

        # z_i in [0, 2i]
        upper = 2.0 * np.arange(1, n + 1, dtype=float)
        super().__init__(int(n), m, 0.0, upper, env)

    def _validate_l(self) -> None:
        """Checks problem-specific constraints on the distance-related variables."""

    @property
    def _degenerate(self) -> bool:
        """Whether A_2 = ... = A_{M-1} = 0 (degenerate front, as in WFG3)."""
        return False

    def _groups(self, t, xp):
        """Splits the position-related variables into M - 1 groups of k / (M - 1)."""
        m = self.n_objectives

        return t[:, : self.k].reshape(t.shape[0], m - 1, self.k // (m - 1))

    def _reduce_uniform(self, t, xp):
        """Uniform weighted-sum reduction to M values (WFG2-8)."""
        position = _r_sum(self._groups(t, xp), None, xp)
        distance = _r_sum(t[:, self.k :], None, xp)

        return xp.concatenate((position, distance[:, None]), axis=1)

    def _reduce_nonsep(self, t, xp):
        """Non-separable reduction to M values (WFG6 and WFG9)."""
        gap = self.k // (self.n_objectives - 1)
        position = _r_nonsep(self._groups(t, xp), gap, xp)
        distance = _r_nonsep(t[:, self.k :], self.l, xp)

        return xp.concatenate((position, distance[:, None]), axis=1)

    def _scales(self, xp, dtype):
        """Scaling constants S_m = 2m, of shape (1, M)."""
        return 2.0 * xp.arange(1, self.n_objectives + 1, dtype=dtype)[None, :]

    def _objectives(self, t, xp):
        """Maps the M reduced parameters to the objective values."""
        tM = t[:, -1:]
        if self._degenerate:
            # A_1 = 1 and A_i = 0 for i > 1
            A = xp.zeros((1, self.n_objectives - 1), dtype=t.dtype)
            A[0, 0] = 1.0
        else:
            A = xp.ones((1, self.n_objectives - 1), dtype=t.dtype)

        x = xp.maximum(tM, A) * (t[:, :-1] - 0.5) + 0.5

        return tM + self._scales(xp, t.dtype) * self._shape(x, xp)

    def _shape(self, x, xp):
        """Shape functions h_1..h_M of the M - 1 position parameters."""
        return _concave(x, xp)

    def _transform(self, y, xp):
        """Transformation chain from the normalized variables to M parameters."""
        raise NotImplementedError

    def _evaluate(self, X, xp):
        return self._objectives(self._transform(X / self._ub, xp), xp)

    def _pareto_front(self, n_points):
        # On the Pareto set x_M = 0 and x_1..x_{M-1} span [0, 1]^{M-1}
        x = _unit_grid(self.n_objectives - 1, n_points)

        return self._scales(np, float) * self._shape(x, np)


class _ConcaveWFG(_WFG):
    """WFG problems whose front is the hypersphere scaled by S (WFG4-9)."""

    def _pareto_front(self, n_points):
        return self._scales(np, float) * _unit_sphere(self.n_objectives, n_points)


class WFG1(_WFG):
    """WFG1: convex and mixed front, flat bias and polynomial bias."""

    def _transform(self, y, xp):
        k = self.k
        distance = _b_flat(_s_linear(y[:, k:], 0.35, xp), 0.8, 0.75, 0.85, xp)
        t = _b_poly(xp.concatenate((y[:, :k], distance), axis=1), 0.02, xp)

        # Weighted sum with weights 2i
        w = 2.0 * xp.arange(1, self.n_variables + 1, dtype=t.dtype)
        position = _r_sum(self._groups(t, xp), self._groups(w[None, :], xp), xp)
        distance = _r_sum(t[:, k:], w[k:], xp)

        return xp.concatenate((position, distance[:, None]), axis=1)

    def _shape(self, x, xp):
        h = _convex(x, xp)
        h[:, -1] = _mixed(x[:, 0], 5.0, 1.0, xp)

        return h

    def _pareto_front(self, n_points):
        return _filtered_front(super()._pareto_front, n_points)


class WFG2(_WFG):
    """WFG2: convex and disconnected front, non-separable distance."""

    def _validate_l(self):
        if self.l % 2:
            raise e.ValueError(f"{self.name} requires an even number of distance variables")

    def _transform(self, y, xp):
        k = self.k
        distance = _s_linear(y[:, k:], 0.35, xp)
        # Pairs of distance-related variables are reduced non-separably
        pairs = distance.reshape(distance.shape[0], self.l // 2, 2)
        t = xp.concatenate((y[:, :k], _r_nonsep(pairs, 2, xp)), axis=1)

        position = _r_sum(self._groups(t, xp), None, xp)
        distance = _r_sum(t[:, k:], None, xp)

        return xp.concatenate((position, distance[:, None]), axis=1)

    def _shape(self, x, xp):
        h = _convex(x, xp)
        h[:, -1] = _disconnected(x[:, 0], 5.0, 1.0, 1.0, xp)

        return h

    def _pareto_front(self, n_points):
        return _filtered_front(super()._pareto_front, n_points)


class WFG3(WFG2):
    """WFG3: linear and degenerate front (a line), non-separable distance."""

    @property
    def _degenerate(self):
        return True

    def _shape(self, x, xp):
        return _linear(x, xp)

    def _pareto_front(self, n_points):
        # Only x_1 varies on the front: x_2 = ... = x_{M-1} = 0.5
        x = np.full((n_points, self.n_objectives - 1), 0.5)
        x[:, 0] = np.linspace(0.0, 1.0, n_points)

        return self._scales(np, float) * self._shape(x, np)


class WFG4(_ConcaveWFG):
    """WFG4: concave front, multi-modal (hill-shaped) landscape."""

    def _transform(self, y, xp):
        return self._reduce_uniform(_s_multi(y, 30.0, 10.0, 0.35, xp), xp)


class WFG5(_ConcaveWFG):
    """WFG5: concave front, deceptive landscape."""

    def _transform(self, y, xp):
        return self._reduce_uniform(_s_decept(y, 0.35, 0.001, 0.05, xp), xp)


class WFG6(_ConcaveWFG):
    """WFG6: concave front, non-separable reduction."""

    def _transform(self, y, xp):
        k = self.k
        t = xp.concatenate((y[:, :k], _s_linear(y[:, k:], 0.35, xp)), axis=1)

        return self._reduce_nonsep(t, xp)


class WFG7(_ConcaveWFG):
    """WFG7: concave front, position parameters biased by the distance ones."""

    def _transform(self, y, xp):
        k = self.k
        u = _suffix_mean(y, xp)[:, :k]
        position = _b_param(y[:, :k], u, 0.98 / 49.98, 0.02, 50.0, xp)
        distance = _s_linear(y[:, k:], 0.35, xp)

        return self._reduce_uniform(xp.concatenate((position, distance), axis=1), xp)


class WFG8(_ConcaveWFG):
    """WFG8: concave front, distance parameters biased by the preceding ones."""

    def _transform(self, y, xp):
        k = self.k
        # u_i = mean(y_1..y_{i-1}) for every distance-related variable i
        u = _prefix_mean(y, xp)[:, k - 1 : -1]
        distance = _b_param(y[:, k:], u, 0.98 / 49.98, 0.02, 50.0, xp)
        distance = _s_linear(distance, 0.35, xp)

        return self._reduce_uniform(xp.concatenate((y[:, :k], distance), axis=1), xp)


class WFG9(_ConcaveWFG):
    """WFG9: concave front, parameter-dependent, deceptive and non-separable."""

    def _transform(self, y, xp):
        k = self.k
        u = _suffix_mean(y, xp)
        biased = _b_param(y[:, :-1], u, 0.98 / 49.98, 0.02, 50.0, xp)
        t = xp.concatenate((biased, y[:, -1:]), axis=1)

        position = _s_decept(t[:, :k], 0.35, 0.001, 0.05, xp)
        distance = _s_multi(t[:, k:], 30.0, 95.0, 0.35, xp)

        return self._reduce_nonsep(xp.concatenate((position, distance), axis=1), xp)
