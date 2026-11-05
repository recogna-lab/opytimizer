"""Shared machinery of the CEC single-objective suites.

It mirrors the organizers' C code: the shift-scale-rotate map (`sr_func`), the
hybrid functions (shift, rotate, shuffle and split into segments), the
composition functions (`cf_cal`) and the basic functions that are not part of
`single_objective`.
"""

import math
from typing import Callable, List, Optional, Sequence

import opytimizer.utils.exception as e
from opytimizer.benchmarks._base import _SingleObjectiveBenchmark
from opytimizer.core.environment import Environment

# Constants of the modified Schwefel's function (official values)
_SCHWEFEL_SHIFT = 420.9687462275036
_SCHWEFEL_OFFSET = 418.9828872724338


def _schaffer_f7(Y, xp):
    """Schaffer's F7 function over consecutive pairs (no wrap-around)."""
    s = xp.sqrt(Y[:, :-1] ** 2 + Y[:, 1:] ** 2)
    total = xp.sum(xp.sqrt(s) * (1.0 + xp.sin(50.0 * s**0.2) ** 2), axis=1)

    return (total / max(1, Y.shape[1] - 1)) ** 2


def _katsuura(Z, xp):
    """Katsuura's function (32 terms per variable)."""
    n = Z.shape[1]
    powers = 2.0 ** xp.arange(1, 33, dtype=Z.dtype)

    # (N, n, 32) terms |2^j z_i - round(2^j z_i)| / 2^j
    T = powers * Z[:, :, None]
    inner = xp.sum(xp.abs(T - xp.floor(T + 0.5)) / powers, axis=2)

    i = xp.arange(1, n + 1, dtype=Z.dtype)
    product = xp.prod((1.0 + i * inner) ** (10.0 / n**1.2), axis=1)

    return 10.0 / n**2 * (product - 1.0)


def _modified_schwefel(Z, xp):
    """Modified Schwefel's function, with a quadratic penalty outside [-500, 500]."""
    n = Z.shape[1]
    t = Z + _SCHWEFEL_SHIFT

    inside = -t * xp.sin(xp.sqrt(xp.abs(t)))

    m_up = xp.fmod(t, 500.0)
    above = -(500.0 - m_up) * xp.sin(xp.sqrt(xp.abs(500.0 - m_up)))
    above = above + ((t - 500.0) / 100.0) ** 2 / n

    m_low = xp.fmod(xp.abs(t), 500.0)
    below = (500.0 - m_low) * xp.sin(xp.sqrt(xp.abs(500.0 - m_low)))
    below = below + ((t + 500.0) / 100.0) ** 2 / n

    g = xp.where(t > 500.0, above, xp.where(t < -500.0, below, inside))

    return xp.sum(g, axis=1) + _SCHWEFEL_OFFSET * n


def _griewank_rosenbrock(Z, xp):
    """Expanded Griewank's plus Rosenbrock's function (with wrap-around)."""
    x = Z + 1.0
    y = xp.roll(x, -1, axis=1)
    t = 100.0 * (x**2 - y) ** 2 + (x - 1.0) ** 2

    return xp.sum(t**2 / 4000.0 - xp.cos(t) + 1.0, axis=1)


def _lunacek_bi_rastrigin(Y, shift, rotation, xp):
    """Lunacek's bi-Rastrigin function, as in the CEC 2017 C code.

    Args:
        Y: Shifted (but not scaled) variables of shape (N, n).
        shift: Shift vector whose signs mirror each variable, of shape (n,).
        rotation: Rotation matrix applied inside the cosine term, or None.
        xp: Array module (numpy / cupy).

    """

    n = Y.shape[1]
    mu0, d = 2.5, 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(n + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0**2 - d) / s)

    # Variables mirrored where the optimum lies on the negative side
    Z = 2.0 * (10.0 / 100.0) * Y * xp.where(shift < 0.0, -1.0, 1.0)

    first = xp.sum(Z**2, axis=1)
    second = s * xp.sum((Z + mu0 - mu1) ** 2, axis=1) + d * n

    R = Z @ rotation.T if rotation is not None else Z

    return xp.minimum(first, second) + 10.0 * (n - xp.sum(xp.cos(2.0 * math.pi * R), axis=1))


def _segment(function: Callable, rate: float = 1.0, offset: float = 0.0) -> Callable:
    """Hybrid part evaluating `function(segment * rate + offset)`."""
    return lambda Y, start, stop, shift, xp: function(Y[:, start:stop] * rate + offset, xp)


class _CECBenchmark(_SingleObjectiveBenchmark):
    """Base class of the CEC single-objective functions.

    Subclasses define `_host_data` (the official data on the host) and
    `_function` (the value without the bias).

    """

    _function_id = 0
    _bias = 0.0
    _dimensions: Sequence[int] = ()

    def __init__(
        self, n_variables: int = 10, env: Optional[Environment] = None
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables, one of the dimensions with
                official data for this function.
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        if n_variables not in self._dimensions:
            raise e.ValueError(
                f"{self.name} is only defined for `n_variables` in {tuple(self._dimensions)}"
            )

        super().__init__(n_variables, env=env)

        # Device copies of the data, created on the first evaluation
        self._device_data = None

    @property
    def optimal_value(self) -> float:
        return self._bias

    def _host_data(self):
        """Official data on the host.

        Returns:
            (Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]): Shifts of shape
                (C, D), rotations of shape (C, D, D) and 0-based shuffles of shape
                (C, D) (None if unused), C being the number of components.

        """

        raise NotImplementedError

    def _optimal_solution(self):
        shift, _, _ = self._host_data()

        return shift[0]

    def _data(self, xp):
        """Shifts, rotations and shuffles on the environment's backend."""
        if self._device_data is None:
            shift, rotation, shuffle = self._host_data()
            self._device_data = (
                xp.asarray(shift, dtype=self.dtype),
                xp.asarray(rotation, dtype=self.dtype),
                None if shuffle is None else xp.asarray(shuffle),
            )

        return self._device_data

    def _shift_rotate(self, X, xp, rate=1.0, rotate=True, component=0):
        """z = M_c ((x - o_c) * rate), the shift-scale-rotate map of the C code."""
        shift, rotation, _ = self._data(xp)
        Z = (X - shift[component]) * rate

        return Z @ rotation[component].T if rotate else Z

    def _hybrid(self, X, xp, fractions: List[float], parts: List[Callable], component=0):
        """Hybrid function: shift, rotate, shuffle and split into segments.

        Args:
            fractions: Fraction of the variables in each segment.
            parts: One callable per segment, mapping (Y, start, stop, shift, xp) to
                its value, where Y is the whole shuffled vector and `shift` the
                shift vector of the hybrid function.
            component: Index of the shift, rotation and shuffle to use.

        """

        shift, _, shuffle = self._data(xp)
        Y = self._shift_rotate(X, xp, component=component)[:, shuffle[component]]

        # ceil(p_i n) variables per segment; the last one takes the remainder
        n = self.n_variables
        sizes = [int(math.ceil(p * n)) for p in fractions[:-1]]
        sizes.append(n - sum(sizes))

        total, start = 0.0, 0
        for size, part in zip(sizes, parts):
            total = total + part(Y, start, start + size, shift[component], xp)
            start += size

        return total

    def _component(self, index, function, rate=1.0, rotate=True, offset=0.0, scale=1.0):
        """Composition part `scale * function(M_i ((x - o_i) rate) + offset)`."""

        def evaluate(X, xp):
            Z = self._shift_rotate(X, xp, rate, rotate, component=index) + offset
            return scale * function(Z, xp)

        return evaluate

    def _composition(self, X, xp, components: List[Callable], sigmas, biases):
        """Composition function: weighted mix of components with their own optima.

        Args:
            components: One callable per component, mapping (X, xp) to its value.
            sigmas: Width of the basin of each component.
            biases: Bias added to each component.

        """

        shift, _, _ = self._data(xp)
        n = self.n_variables

        F = xp.stack(
            [component(X, xp) + bias for component, bias in zip(components, biases)],
            axis=1,
        )
        D = xp.stack(
            [xp.sum((X - shift[i]) ** 2, axis=1) for i in range(len(components))],
            axis=1,
        )
        sigma = xp.asarray(sigmas, dtype=D.dtype)

        # w_i = exp(-d_i / (2 n sigma_i^2)) / sqrt(d_i), d_i the squared distance
        safe = xp.where(D == 0.0, 1.0, D)
        W = xp.exp(-safe / (2.0 * n * sigma**2)) / xp.sqrt(safe)

        # At a component's optimum its weight dominates (the C code yields a huge
        # or infinite weight there)
        zero = D == 0.0
        W = xp.where(xp.any(zero, axis=1, keepdims=True), zero.astype(W.dtype), W)

        # Every weight underflowed: the components are averaged
        W = xp.where(xp.sum(W, axis=1, keepdims=True) == 0.0, 1.0, W)

        return xp.sum(W * F, axis=1) / xp.sum(W, axis=1)

    def _function(self, X, xp):
        """Value of the function without the bias."""
        raise NotImplementedError

    def _evaluate(self, X, xp):
        return self._function(X, xp) + self._bias
