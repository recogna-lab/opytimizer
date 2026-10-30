"""ZDT family of bi-objective benchmark problems.

ZDT5 is not included, since it is defined over binary strings.

References:
    References

Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective evolutionary
algorithms: empirical results. Evolutionary Computation, 8(2), 173-195.
doi:10.1162/106365600568202

"""

import math
from typing import Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.benchmarks._base import _MultiObjectiveBenchmark
from opytimizer.benchmarks._utils import _filtered_front
from opytimizer.core.environment import Environment


class _ZDT(_MultiObjectiveBenchmark):
    """Base class of the ZDT problems: f_1 = f_1(x_1), f_2 = g(x_2..x_n) h(f_1, g)."""

    # Number of variables recommended by Zitzler et al. (2000)
    _default_n_variables = 30

    def __init__(
        self, n_variables: Optional[int] = None, env: Optional[Environment] = None
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables (at least 2). Defaults to the
                value recommended in the literature (30, or 10 for ZDT4 and ZDT6).
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        n = self._default_n_variables if n_variables is None else n_variables
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise e.ValueError("ZDT problems require `n_variables` >= 2")

        lb, ub = self._bounds(int(n))
        super().__init__(int(n), 2, lb, ub, env)

    def _bounds(self, n: int):
        """Lower and upper bounds of the decision variables."""
        return 0.0, 1.0

    def _f1(self, X, xp):
        """First objective, which depends on x_1 only."""
        return X[:, 0]

    def _g(self, X, xp):
        """Distance function g >= 1 (g = 1 on the Pareto set)."""
        return 1.0 + 9.0 * xp.sum(X[:, 1:], axis=1) / (X.shape[1] - 1)

    def _h(self, f1, g, xp):
        """Shape function of the second objective."""
        return 1.0 - xp.sqrt(f1 / g)

    def _evaluate(self, X, xp):
        f1 = self._f1(X, xp)
        g = self._g(X, xp)

        return xp.stack((f1, g * self._h(f1, g, xp)), axis=1)

    def _f1_range(self):
        """Range of f_1 over the Pareto set."""
        return 0.0, 1.0

    def _pareto_front(self, n_points):
        # On the Pareto set g = 1, so the front is f_2 = h(f_1, 1)
        f1 = np.linspace(*self._f1_range(), n_points)

        return np.stack((f1, self._h(f1, np.ones_like(f1), np)), axis=1)


class ZDT1(_ZDT):
    """ZDT1: convex front, f_2 = 1 - sqrt(f_1)."""


class ZDT2(_ZDT):
    """ZDT2: concave front, f_2 = 1 - f_1^2."""

    def _h(self, f1, g, xp):
        return 1.0 - (f1 / g) ** 2


class ZDT3(_ZDT):
    """ZDT3: front made of five disconnected convex parts."""

    def _h(self, f1, g, xp):
        return 1.0 - xp.sqrt(f1 / g) - (f1 / g) * xp.sin(10.0 * math.pi * f1)

    def _pareto_front(self, n_points):
        # h is not monotonic: only the non-dominated parts of the curve remain
        return _filtered_front(super()._pareto_front, n_points)


class ZDT4(_ZDT):
    """ZDT4: convex front with 21^9 local fronts (Rastrigin-like g)."""

    _default_n_variables = 10

    def _bounds(self, n):
        return [0.0] + [-5.0] * (n - 1), [1.0] + [5.0] * (n - 1)

    def _g(self, X, xp):
        Y = X[:, 1:]

        return (
            1.0
            + 10.0 * Y.shape[1]
            + xp.sum(Y**2 - 10.0 * xp.cos(4.0 * math.pi * Y), axis=1)
        )


class ZDT6(_ZDT):
    """ZDT6: concave front with a non-uniform density of solutions."""

    _default_n_variables = 10

    def _f1(self, X, xp):
        x1 = X[:, 0]

        return 1.0 - xp.exp(-4.0 * x1) * xp.sin(6.0 * math.pi * x1) ** 6

    def _g(self, X, xp):
        return 1.0 + 9.0 * (xp.sum(X[:, 1:], axis=1) / (X.shape[1] - 1)) ** 0.25

    def _h(self, f1, g, xp):
        return 1.0 - (f1 / g) ** 2

    def _f1_range(self):
        # Minimum of 1 - exp(-4 x) sin^6(6 pi x) over [0, 1]
        return 0.2807753191, 1.0
