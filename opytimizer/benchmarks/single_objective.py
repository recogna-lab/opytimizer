"""Single-objective benchmark functions.

Classic n-dimensional functions widely adopted in the literature (and as base
functions of the CEC suites), all in their standard, non-shifted form.

References:
    M. Jamil and X.-S. Yang, "A literature survey of benchmark functions for
    global optimization problems," Int. J. Math. Model. Numer. Optim., vol. 4,
    no. 2, pp. 150-194, 2013.

    S. Surjanovic and D. Bingham, "Virtual Library of Simulation Experiments:
    Test Functions and Datasets," Simon Fraser University, 2013.

    A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi, and P. N. Suganthan,
    "Problem definitions and evaluation criteria for the CEC 2022 special session
    and competition on single objective bound constrained numerical
    optimization," Technical Report, 2021.

"""

import math
from typing import Any

import numpy as np

from opytimizer.benchmarks._base import _SingleObjectiveBenchmark


def _indices(X: Any, xp: Any) -> Any:
    """1-based variable indices, of shape (1, n_variables)."""
    return xp.arange(1, X.shape[1] + 1, dtype=X.dtype)[None, :]


class Sphere(_SingleObjectiveBenchmark):
    """Sphere: f(x) = sum(x_i^2). Unimodal, separable; f* = 0 at x* = 0."""

    def _evaluate(self, X, xp):
        return xp.sum(X**2, axis=1)


class SumSquares(_SingleObjectiveBenchmark):
    """Sum of squares: f(x) = sum(i * x_i^2). Unimodal; f* = 0 at x* = 0."""

    _default_bounds = (-10.0, 10.0)

    def _evaluate(self, X, xp):
        return xp.sum(_indices(X, xp) * X**2, axis=1)


def _elliptic(X, xp):
    """High conditioned elliptic function, evaluated row-wise on a batch of shape (N, n)."""
    n = X.shape[1]
    exponent = (_indices(X, xp) - 1.0) / max(1, n - 1)

    return xp.sum((1.0e6**exponent) * X**2, axis=1)


class HighConditionedElliptic(_SingleObjectiveBenchmark):
    """High conditioned elliptic: f(x) = sum((10^6)^((i-1)/(n-1)) * x_i^2).

    Unimodal, ill-conditioned (CEC base function); f* = 0 at x* = 0.
    """

    def _evaluate(self, X, xp):
        return _elliptic(X, xp)


def _bent_cigar(X, xp):
    """Bent cigar function, evaluated row-wise on a batch of shape (N, n)."""
    return X[:, 0] ** 2 + 1.0e6 * xp.sum(X[:, 1:] ** 2, axis=1)


class BentCigar(_SingleObjectiveBenchmark):
    """Bent cigar: f(x) = x_1^2 + 10^6 * sum_{i>1}(x_i^2). f* = 0 at x* = 0."""

    def _evaluate(self, X, xp):
        return _bent_cigar(X, xp)


def _discus(X, xp):
    """Discus function, evaluated row-wise on a batch of shape (N, n)."""
    return 1.0e6 * X[:, 0] ** 2 + xp.sum(X[:, 1:] ** 2, axis=1)


class Discus(_SingleObjectiveBenchmark):
    """Discus: f(x) = 10^6 * x_1^2 + sum_{i>1}(x_i^2). f* = 0 at x* = 0."""

    def _evaluate(self, X, xp):
        return _discus(X, xp)


class Quartic(_SingleObjectiveBenchmark):
    """Quartic (noiseless): f(x) = sum(i * x_i^4). f* = 0 at x* = 0."""

    _default_bounds = (-1.28, 1.28)

    def _evaluate(self, X, xp):
        return xp.sum(_indices(X, xp) * X**4, axis=1)


class Step(_SingleObjectiveBenchmark):
    """Step: f(x) = sum(floor(x_i + 0.5)^2). Discontinuous; f* = 0 on [-0.5, 0.5)^n."""

    def _evaluate(self, X, xp):
        return xp.sum(xp.floor(X + 0.5) ** 2, axis=1)


class SchwefelProblem12(_SingleObjectiveBenchmark):
    """Schwefel's problem 1.2: f(x) = sum_i (sum_{j<=i} x_j)^2. f* = 0 at x* = 0."""

    def _evaluate(self, X, xp):
        return xp.sum(xp.cumsum(X, axis=1) ** 2, axis=1)


class SchwefelProblem222(_SingleObjectiveBenchmark):
    """Schwefel's problem 2.22: f(x) = sum|x_i| + prod|x_i|. f* = 0 at x* = 0."""

    _default_bounds = (-10.0, 10.0)

    def _evaluate(self, X, xp):
        A = xp.abs(X)

        return xp.sum(A, axis=1) + xp.prod(A, axis=1)


def _zakharov(X, xp):
    """Zakharov function, evaluated row-wise on a batch of shape (N, n)."""
    s = xp.sum(0.5 * _indices(X, xp) * X, axis=1)

    return xp.sum(X**2, axis=1) + s**2 + s**4


class Zakharov(_SingleObjectiveBenchmark):
    """Zakharov: f(x) = sum(x_i^2) + s^2 + s^4, s = sum(0.5 * i * x_i). f* = 0 at 0."""

    _default_bounds = (-5.0, 10.0)

    def _evaluate(self, X, xp):
        return _zakharov(X, xp)


def _rosenbrock(X, xp):
    """Rosenbrock function, evaluated row-wise on a batch of shape (N, n)."""
    a, b = X[:, :-1], X[:, 1:]

    return xp.sum(100.0 * (b - a**2) ** 2 + (a - 1.0) ** 2, axis=1)


class Rosenbrock(_SingleObjectiveBenchmark):
    """Rosenbrock: f(x) = sum(100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2). f* = 0 at x* = 1."""

    _default_bounds = (-5.0, 10.0)

    def _evaluate(self, X, xp):
        return _rosenbrock(X, xp)

    def _optimal_solution(self):
        return 1.0


class DixonPrice(_SingleObjectiveBenchmark):
    """Dixon-Price: f(x) = (x_1 - 1)^2 + sum_{i>1} i (2 x_i^2 - x_{i-1})^2. f* = 0."""

    _default_bounds = (-10.0, 10.0)

    def _evaluate(self, X, xp):
        i = _indices(X, xp)[:, 1:]

        return (X[:, 0] - 1.0) ** 2 + xp.sum(
            i * (2.0 * X[:, 1:] ** 2 - X[:, :-1]) ** 2, axis=1
        )

    def _optimal_solution(self):
        # x*_i = 2^(-(2^i - 2) / 2^i)
        i = np.arange(1, self.n_variables + 1, dtype=float)

        return 2.0 ** (-(2.0**i - 2.0) / 2.0**i)


def _rastrigin(X, xp):
    """Rastrigin function, evaluated row-wise on a batch of shape (N, n)."""
    return 10.0 * X.shape[1] + xp.sum(X**2 - 10.0 * xp.cos(2.0 * math.pi * X), axis=1)


class Rastrigin(_SingleObjectiveBenchmark):
    """Rastrigin: f(x) = 10 n + sum(x_i^2 - 10 cos(2 pi x_i)). Multimodal; f* = 0 at 0."""

    _default_bounds = (-5.12, 5.12)

    def _evaluate(self, X, xp):
        return _rastrigin(X, xp)


def _ackley(X, xp):
    """Ackley function, evaluated row-wise on a batch of shape (N, n)."""
    n = X.shape[1]
    square = xp.sqrt(xp.sum(X**2, axis=1) / n)
    cosine = xp.sum(xp.cos(2.0 * math.pi * X), axis=1) / n

    return -20.0 * xp.exp(-0.2 * square) - xp.exp(cosine) + 20.0 + math.e


class Ackley(_SingleObjectiveBenchmark):
    """Ackley (a = 20, b = 0.2, c = 2 pi). Multimodal; f* = 0 at x* = 0."""

    _default_bounds = (-32.768, 32.768)

    def _evaluate(self, X, xp):
        return _ackley(X, xp)


def _griewank(X, xp):
    """Griewank function, evaluated row-wise on a batch of shape (N, n)."""
    cosine = xp.prod(xp.cos(X / xp.sqrt(_indices(X, xp))), axis=1)

    return 1.0 + xp.sum(X**2, axis=1) / 4000.0 - cosine


class Griewank(_SingleObjectiveBenchmark):
    """Griewank: f(x) = 1 + sum(x_i^2) / 4000 - prod(cos(x_i / sqrt(i))). f* = 0 at 0."""

    _default_bounds = (-600.0, 600.0)

    def _evaluate(self, X, xp):
        return _griewank(X, xp)


class Schwefel(_SingleObjectiveBenchmark):
    """Schwefel's problem 2.26: f(x) = 418.9829 n - sum(x_i sin(sqrt|x_i|)).

    Deceptive multimodal function; f* ~ 0 at x* ~ 420.9687.
    """

    _default_bounds = (-500.0, 500.0)

    def _evaluate(self, X, xp):
        return 418.9828872724338 * X.shape[1] - xp.sum(
            X * xp.sin(xp.sqrt(xp.abs(X))), axis=1
        )

    def _optimal_solution(self):
        return 420.968746359982


def _levy(X, xp):
    """Levy function, evaluated row-wise on a batch of shape (N, n)."""
    w = 1.0 + (X - 1.0) / 4.0
    head = xp.sin(math.pi * w[:, 0]) ** 2
    body = xp.sum(
        (w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * xp.sin(math.pi * w[:, :-1] + 1.0) ** 2),
        axis=1,
    )
    tail = (w[:, -1] - 1.0) ** 2 * (1.0 + xp.sin(2.0 * math.pi * w[:, -1]) ** 2)

    return head + body + tail


class Levy(_SingleObjectiveBenchmark):
    """Levy: w_i = 1 + (x_i - 1) / 4. Multimodal; f* = 0 at x* = 1."""

    _default_bounds = (-10.0, 10.0)

    def _evaluate(self, X, xp):
        return _levy(X, xp)

    def _optimal_solution(self):
        return 1.0


class StyblinskiTang(_SingleObjectiveBenchmark):
    """Styblinski-Tang: f(x) = 0.5 sum(x_i^4 - 16 x_i^2 + 5 x_i).

    f* = -39.16616570377142 n at x*_i = -2.903534027771178.
    """

    _default_bounds = (-5.0, 5.0)

    def _evaluate(self, X, xp):
        return 0.5 * xp.sum(X**4 - 16.0 * X**2 + 5.0 * X, axis=1)

    @property
    def optimal_value(self):
        return -39.16616570377142 * self.n_variables

    def _optimal_solution(self):
        return -2.903534027771178


class Alpine(_SingleObjectiveBenchmark):
    """Alpine N. 1: f(x) = sum|x_i sin(x_i) + 0.1 x_i|. f* = 0 at x* = 0."""

    _default_bounds = (-10.0, 10.0)

    def _evaluate(self, X, xp):
        return xp.sum(xp.abs(X * xp.sin(X) + 0.1 * X), axis=1)


class Salomon(_SingleObjectiveBenchmark):
    """Salomon: f(x) = 1 - cos(2 pi ||x||) + 0.1 ||x||. f* = 0 at x* = 0."""

    def _evaluate(self, X, xp):
        norm = xp.sqrt(xp.sum(X**2, axis=1))

        return 1.0 - xp.cos(2.0 * math.pi * norm) + 0.1 * norm


class Weierstrass(_SingleObjectiveBenchmark):
    """Weierstrass (a = 0.5, b = 3, k_max = 20). Continuous, nowhere differentiable.

    f* = 0 at x* = 0.
    """

    _default_bounds = (-0.5, 0.5)

    def _evaluate(self, X, xp):
        k = xp.arange(21, dtype=X.dtype)
        a_k, b_k = 0.5**k, 3.0**k

        # (N, n, k_max + 1) terms a^k cos(2 pi b^k (x_i + 0.5))
        terms = a_k * xp.cos(2.0 * math.pi * b_k * (X[:, :, None] + 0.5))
        offset = X.shape[1] * xp.sum(a_k * xp.cos(math.pi * b_k))

        return xp.sum(terms, axis=(1, 2)) - offset


def _expanded_schaffer_f6(X, xp):
    """Expanded Schaffer's F6 function, evaluated row-wise on a batch of shape (N, n)."""
    # Pairs (x_i, x_{i+1}), wrapping around (x_n, x_1)
    s = X**2 + xp.roll(X, -1, axis=1) ** 2

    return xp.sum(
        0.5 + (xp.sin(xp.sqrt(s)) ** 2 - 0.5) / (1.0 + 0.001 * s) ** 2, axis=1
    )


class ExpandedSchafferF6(_SingleObjectiveBenchmark):
    """Expanded Schaffer's F6 (CEC base function), summed over consecutive pairs.

    f* = 0 at x* = 0.
    """

    def _evaluate(self, X, xp):
        return _expanded_schaffer_f6(X, xp)


def _happycat(X, xp):
    """HappyCat function, evaluated row-wise on a batch of shape (N, n)."""
    n = X.shape[1]
    r2 = xp.sum(X**2, axis=1)
    s = xp.sum(X, axis=1)

    return xp.abs(r2 - n) ** 0.25 + (0.5 * r2 + s) / n + 0.5


class HappyCat(_SingleObjectiveBenchmark):
    """HappyCat (CEC base function). f* = 0 at x* = -1."""

    _default_bounds = (-2.0, 2.0)

    def _evaluate(self, X, xp):
        return _happycat(X, xp)

    def _optimal_solution(self):
        return -1.0


def _hgbat(X, xp):
    """HGBat function, evaluated row-wise on a batch of shape (N, n)."""
    n = X.shape[1]
    r2 = xp.sum(X**2, axis=1)
    s = xp.sum(X, axis=1)

    return xp.abs(r2**2 - s**2) ** 0.5 + (0.5 * r2 + s) / n + 0.5


class HGBat(_SingleObjectiveBenchmark):
    """HGBat (CEC base function). f* = 0 at x* = -1."""

    _default_bounds = (-2.0, 2.0)

    def _evaluate(self, X, xp):
        return _hgbat(X, xp)

    def _optimal_solution(self):
        return -1.0
