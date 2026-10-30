"""DTLZ family of scalable multi- and many-objective benchmark problems.

The first M - 1 variables define the position on the front and the remaining
k = n - M + 1 variables (x_M) define the distance g(x_M) to it.

References:
    Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005). Scalable test problems
    for evolutionary multiobjective optimization. In Advanced Information and Knowledge
    Processing (pp. 105-145). doi:10.1007/1-84628-137-7_6

"""

import math
from typing import Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.benchmarks._base import _MultiObjectiveBenchmark
from opytimizer.benchmarks._utils import (
    _filtered_front,
    _product_shape,
    _simplex_lattice,
    _unit_grid,
    _unit_sphere,
)
from opytimizer.core.environment import Environment


def _g_multimodal(XM, xp):
    """Rastrigin-like distance of DTLZ1 and DTLZ3 (11^k - 1 local fronts)."""
    return 100.0 * (
        XM.shape[1]
        + xp.sum((XM - 0.5) ** 2 - xp.cos(20.0 * math.pi * (XM - 0.5)), axis=1)
    )


def _g_sphere(XM, xp):
    """Unimodal distance of DTLZ2, DTLZ4 and DTLZ5."""
    return xp.sum((XM - 0.5) ** 2, axis=1)


def _spherical(theta, g, xp):
    """Spherical front of radius 1 + g parameterized by the angles `theta`."""
    return (1.0 + g)[:, None] * _product_shape(xp.cos(theta), xp.sin(theta), xp)


class _DTLZ(_MultiObjectiveBenchmark):
    """Base class of the DTLZ problems."""

    # Size of x_M recommended by Deb et al. (2005)
    _default_k = 10

    def __init__(
        self,
        n_variables: Optional[int] = None,
        n_objectives: int = 3,
        env: Optional[Environment] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables (at least `n_objectives`).
                Defaults to n_objectives + k - 1, with the k recommended in the
                literature (5 for DTLZ1, 10 for DTLZ2-6 and 20 for DTLZ7).
            n_objectives: Number of objectives (at least 2).
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        if not isinstance(n_objectives, (int, np.integer)) or n_objectives < 2:
            raise e.ValueError("DTLZ problems require `n_objectives` >= 2")

        n = n_objectives + self._default_k - 1 if n_variables is None else n_variables
        if not isinstance(n, (int, np.integer)) or n < n_objectives:
            raise e.ValueError("DTLZ problems require `n_variables` >= `n_objectives`")

        super().__init__(int(n), int(n_objectives), 0.0, 1.0, env)

    def _split(self, X):
        """Splits the variables into position (x_1..x_{M-1}) and distance (x_M) ones."""
        m = self.n_objectives

        return X[:, : m - 1], X[:, m - 1 :]


class DTLZ1(_DTLZ):
    """DTLZ1: linear front sum(f) = 0.5 with 11^k - 1 local fronts."""

    _default_k = 5

    def _evaluate(self, X, xp):
        XI, XM = self._split(X)
        g = _g_multimodal(XM, xp)

        return 0.5 * (1.0 + g)[:, None] * _product_shape(XI, 1.0 - XI, xp)

    def _pareto_front(self, n_points):
        return 0.5 * _simplex_lattice(self.n_objectives, n_points)


class DTLZ2(_DTLZ):
    """DTLZ2: concave front on the unit hypersphere."""

    def _evaluate(self, X, xp):
        XI, XM = self._split(X)

        return _spherical(0.5 * math.pi * XI, _g_sphere(XM, xp), xp)

    def _pareto_front(self, n_points):
        return _unit_sphere(self.n_objectives, n_points)


class DTLZ3(DTLZ2):
    """DTLZ3: DTLZ2 front with the multimodal distance of DTLZ1 (3^k - 1 local fronts)."""

    def _evaluate(self, X, xp):
        XI, XM = self._split(X)

        return _spherical(0.5 * math.pi * XI, _g_multimodal(XM, xp), xp)


class DTLZ4(DTLZ2):
    """DTLZ4: DTLZ2 with a biased density of solutions (x_i^alpha, alpha = 100)."""

    _alpha = 100.0

    def _evaluate(self, X, xp):
        XI, XM = self._split(X)

        return _spherical(0.5 * math.pi * XI**self._alpha, _g_sphere(XM, xp), xp)


class DTLZ5(_DTLZ):
    """DTLZ5: degenerate front, a curve on the unit hypersphere."""

    def _distance(self, XM, xp):
        """Distance function g of the problem."""
        return _g_sphere(XM, xp)

    def _evaluate(self, X, xp):
        XI, XM = self._split(X)
        g = self._distance(XM, xp)

        # theta_1 = x_1 pi / 2 and theta_i = pi (1 + 2 g x_i) / (4 (1 + g)), i > 1
        theta = math.pi / (4.0 * (1.0 + g[:, None])) * (1.0 + 2.0 * g[:, None] * XI)
        theta = xp.concatenate((0.5 * math.pi * XI[:, :1], theta[:, 1:]), axis=1)

        return _spherical(theta, g, xp)

    def _pareto_front(self, n_points):
        # With g = 0 every angle but the first one is pi / 4
        m = self.n_objectives
        theta = np.full((n_points, m - 1), 0.25 * math.pi)
        theta[:, 0] = np.linspace(0.0, 0.5 * math.pi, n_points)

        return _spherical(theta, np.zeros(n_points), np)


class DTLZ6(DTLZ5):
    """DTLZ6: DTLZ5 with a harder distance function g = sum(x_i^0.1)."""

    def _distance(self, XM, xp):
        return xp.sum(XM**0.1, axis=1)


class DTLZ7(_DTLZ):
    """DTLZ7: front with 2^(M-1) disconnected regions."""

    _default_k = 20

    def _h(self, FI, g, xp):
        """Shape function of the last objective."""
        m = self.n_objectives

        return m - xp.sum(
            FI / (1.0 + g[:, None]) * (1.0 + xp.sin(3.0 * math.pi * FI)), axis=1
        )

    def _evaluate(self, X, xp):
        XI, XM = self._split(X)
        g = 1.0 + 9.0 * xp.sum(XM, axis=1) / XM.shape[1]

        fM = (1.0 + g) * self._h(XI, g, xp)

        return xp.concatenate((XI, fM[:, None]), axis=1)

    def _front_image(self, n_points):
        """Image of the Pareto set (g = 1), including its dominated regions."""
        FI = _unit_grid(self.n_objectives - 1, n_points)
        g = np.ones(FI.shape[0])
        fM = (1.0 + g) * self._h(FI, g, np)

        return np.concatenate((FI, fM[:, None]), axis=1)

    def _pareto_front(self, n_points):
        # Only the non-dominated regions of the image remain
        return _filtered_front(self._front_image, n_points)
