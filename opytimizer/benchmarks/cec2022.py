"""CEC 2022 single-objective bound-constrained benchmark suite (F1-F12).

The functions follow the organizers' reference C code (`cec22_test_func.cpp`),
which is what the published results of the competition were computed with.
Where that code differs from the technical report, the code is followed:

    - F3 evaluates Schaffer's F7 on the shifted (unrotated) variables.
    - F4 is the shifted and rotated Rastrigin's function: the rounding of the
      non-continuous variant is applied to a buffer the code then overwrites.
    - Inside F7, Schaffer's F7 reads the first variables of the shuffled vector
      instead of its own segment.

The only deviation from the C code is at a composition optimum: the C code
computes inf / inf there, whereas here the limit (the optimum's bias) is used.

Every function is defined on [-100, 100]^D for D in {2, 10, 20} (hybrid
functions F6-F8 only for D in {10, 20}), keeps the official biases and has its
global minimum f* = bias at the first shift vector. The shift vectors, rotation
matrices and shuffles ship with the package and are only read when a problem is
first evaluated.

References:
    A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi, and P. N. Suganthan,
    "Problem definitions and evaluation criteria for the CEC 2022 special session
    and competition on single objective bound constrained numerical
    optimization," Technical Report, Nanyang Technological University, 2021.

"""

import functools
import math
from pathlib import Path
from typing import Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.benchmarks._base import _SingleObjectiveBenchmark
from opytimizer.benchmarks.single_objective import (
    _ackley,
    _bent_cigar,
    _discus,
    _elliptic,
    _expanded_schaffer_f6,
    _griewank,
    _happycat,
    _hgbat,
    _levy,
    _rastrigin,
    _rosenbrock,
    _zakharov,
)
from opytimizer.core.environment import Environment

_DATA_DIR = Path(__file__).resolve().parent / "data" / "cec2022"

# Constants of the modified Schwefel's function (official values)
_SCHWEFEL_SHIFT = 420.9687462275036
_SCHWEFEL_OFFSET = 418.9828872724338


@functools.lru_cache(maxsize=None)
def _load_data(function_id: int, n_variables: int):
    """Reads the shift vectors, rotation matrices and shuffle of a function.

    Returns:
        (Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]): Shifts of shape
            (C, D), rotations of shape (C, D, D) and the 0-based shuffle of shape
            (D,) (only for the hybrid functions), C being the number of components.

    """

    n = n_variables
    shift = np.loadtxt(_DATA_DIR / f"shift_data_{function_id}.txt", ndmin=2)[:, :n]
    rotation = np.loadtxt(_DATA_DIR / f"M_{function_id}_D{n}.txt").reshape(-1, n, n)

    shuffle = None
    if 6 <= function_id <= 8:
        path = _DATA_DIR / f"shuffle_data_{function_id}_D{n}.txt"
        shuffle = np.loadtxt(path, dtype=np.int64).reshape(-1)[:n] - 1

    return shift, rotation, shuffle


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


class _CEC2022(_SingleObjectiveBenchmark):
    """Base class of the CEC 2022 functions."""

    _function_id = 0
    _bias = 0.0
    _dimensions = (2, 10, 20)

    def __init__(
        self, n_variables: int = 10, env: Optional[Environment] = None
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables, one of the dimensions with
                official data (2, 10 or 20; 10 or 20 for F6-F8).
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        if n_variables not in self._dimensions:
            raise e.ValueError(
                f"{self.name} is only defined for `n_variables` in {self._dimensions}"
            )

        super().__init__(n_variables, env=env)

        # Device copies of the data, created on the first evaluation
        self._device_data = None

    @property
    def optimal_value(self) -> float:
        return self._bias

    def _optimal_solution(self):
        shift, _, _ = _load_data(self._function_id, self.n_variables)

        return shift[0]

    def _data(self, xp):
        """Shifts, rotations and shuffle on the environment's backend."""
        if self._device_data is None:
            shift, rotation, shuffle = _load_data(self._function_id, self.n_variables)
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

    def _hybrid(self, X, xp, fractions, parts):
        """Hybrid function: shift, rotate, shuffle and split into segments.

        Args:
            fractions: Fraction of the variables in each segment.
            parts: One callable per segment, mapping (Y, start, stop) to its value,
                where Y is the whole shuffled vector.

        """

        _, _, shuffle = self._data(xp)
        Y = self._shift_rotate(X, xp)[:, shuffle]

        # ceil(p_i n) variables per segment; the last one takes the remainder
        n = self.n_variables
        sizes = [int(math.ceil(p * n)) for p in fractions[:-1]]
        sizes.append(n - sum(sizes))

        total, start = 0.0, 0
        for size, part in zip(sizes, parts):
            total = total + part(Y, start, start + size, xp)
            start += size

        return total

    def _composition(self, X, xp, components, sigmas, biases):
        """Composition function: weighted mix of shifted and rotated components.

        Args:
            components: One (function, rate, rotate, offset, scale) tuple per
                component, giving scale * function(M_i ((x - o_i) rate) + offset).
            sigmas: Width of the basin of each component.
            biases: Bias added to each component.

        """

        shift, _, _ = self._data(xp)
        n = self.n_variables

        fits, distances = [], []
        for i, (function, rate, rotate, offset, scale) in enumerate(components):
            Z = self._shift_rotate(X, xp, rate, rotate, component=i) + offset
            fits.append(scale * function(Z, xp) + biases[i])
            distances.append(xp.sum((X - shift[i]) ** 2, axis=1))

        F = xp.stack(fits, axis=1)
        D = xp.stack(distances, axis=1)
        sigma = xp.asarray(sigmas, dtype=D.dtype)

        # w_i = exp(-d_i / (2 n sigma_i^2)) / sqrt(d_i), d_i the squared distance
        safe = xp.where(D == 0.0, 1.0, D)
        W = xp.exp(-safe / (2.0 * n * sigma**2)) / xp.sqrt(safe)

        # At a component's optimum its weight dominates (the C code yields inf / inf)
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


def _segment(function, rate=1.0, offset=0.0):
    """Hybrid part evaluating `function(segment * rate + offset)`."""
    return lambda Y, start, stop, xp: function(Y[:, start:stop] * rate + offset, xp)


class CEC2022F1(_CEC2022):
    """CEC 2022 F1: shifted and full rotated Zakharov's function (f* = 300)."""

    _function_id = 1
    _bias = 300.0

    def _function(self, X, xp):
        return _zakharov(self._shift_rotate(X, xp), xp)


class CEC2022F2(_CEC2022):
    """CEC 2022 F2: shifted and full rotated Rosenbrock's function (f* = 400)."""

    _function_id = 2
    _bias = 400.0

    def _function(self, X, xp):
        return _rosenbrock(self._shift_rotate(X, xp, 2.048 / 100.0) + 1.0, xp)


class CEC2022F3(_CEC2022):
    """CEC 2022 F3: shifted Schaffer's F7 function (f* = 600).

    The report lists a rotated expanded Schaffer's f6; the official code evaluates
    Schaffer's F7 on the shifted variables, before the rotation.
    """

    _function_id = 3
    _bias = 600.0

    def _function(self, X, xp):
        return _schaffer_f7(self._shift_rotate(X, xp, rotate=False), xp)


class CEC2022F4(_CEC2022):
    """CEC 2022 F4: shifted and full rotated (non-continuous) Rastrigin's function.

    As in the official code, the rounding step has no effect (f* = 800).
    """

    _function_id = 4
    _bias = 800.0

    def _function(self, X, xp):
        return _rastrigin(self._shift_rotate(X, xp, 5.12 / 100.0), xp)


class CEC2022F5(_CEC2022):
    """CEC 2022 F5: shifted and full rotated Levy's function (f* = 900)."""

    _function_id = 5
    _bias = 900.0

    def _function(self, X, xp):
        # The C code uses w = 1 + z / 4, i.e., Levy's function of z + 1
        return _levy(self._shift_rotate(X, xp) + 1.0, xp)


class CEC2022F6(_CEC2022):
    """CEC 2022 F6: hybrid function 1 (Bent Cigar, HGBat, Rastrigin; f* = 1800)."""

    _function_id = 6
    _bias = 1800.0
    _dimensions = (10, 20)

    def _function(self, X, xp):
        parts = [
            _segment(_bent_cigar),
            _segment(_hgbat, 5.0 / 100.0, -1.0),
            _segment(_rastrigin, 5.12 / 100.0),
        ]

        return self._hybrid(X, xp, [0.4, 0.4, 0.2], parts)


class CEC2022F7(_CEC2022):
    """CEC 2022 F7: hybrid function 2 (HGBat, Katsuura, Ackley, Rastrigin,
    modified Schwefel, Schaffer's F7; f* = 2000).
    """

    _function_id = 7
    _bias = 2000.0
    _dimensions = (10, 20)

    def _function(self, X, xp):
        parts = [
            _segment(_hgbat, 5.0 / 100.0, -1.0),
            _segment(_katsuura, 5.0 / 100.0),
            _segment(_ackley),
            _segment(_rastrigin, 5.12 / 100.0),
            _segment(_modified_schwefel, 1000.0 / 100.0),
            # The C code reads the first `stop - start` shuffled variables here
            lambda Y, start, stop, xp: _schaffer_f7(Y[:, : stop - start], xp),
        ]

        return self._hybrid(X, xp, [0.1, 0.2, 0.2, 0.2, 0.1, 0.2], parts)


class CEC2022F8(_CEC2022):
    """CEC 2022 F8: hybrid function 3 (Katsuura, HappyCat, Griewank-Rosenbrock,
    modified Schwefel, Ackley; f* = 2200).
    """

    _function_id = 8
    _bias = 2200.0
    _dimensions = (10, 20)

    def _function(self, X, xp):
        parts = [
            _segment(_katsuura, 5.0 / 100.0),
            _segment(_happycat, 5.0 / 100.0, -1.0),
            _segment(_griewank_rosenbrock, 5.0 / 100.0),
            _segment(_modified_schwefel, 1000.0 / 100.0),
            _segment(_ackley),
        ]

        return self._hybrid(X, xp, [0.3, 0.2, 0.2, 0.1, 0.2], parts)


class CEC2022F9(_CEC2022):
    """CEC 2022 F9: composition function 1 (Rosenbrock, elliptic, Bent Cigar,
    Discus, elliptic; f* = 2300).
    """

    _function_id = 9
    _bias = 2300.0

    def _function(self, X, xp):
        components = [
            (_rosenbrock, 2.048 / 100.0, True, 1.0, 1.0),
            (_elliptic, 1.0, True, 0.0, 1.0e-6),
            (_bent_cigar, 1.0, True, 0.0, 1.0e-26),
            (_discus, 1.0, True, 0.0, 1.0e-6),
            (_elliptic, 1.0, False, 0.0, 1.0e-6),
        ]

        return self._composition(
            X, xp, components, [10, 20, 30, 40, 50], [0, 200, 300, 100, 400]
        )


class CEC2022F10(_CEC2022):
    """CEC 2022 F10: composition function 2 (modified Schwefel, Rastrigin, HGBat;
    f* = 2400).
    """

    _function_id = 10
    _bias = 2400.0

    def _function(self, X, xp):
        components = [
            (_modified_schwefel, 1000.0 / 100.0, False, 0.0, 1.0),
            (_rastrigin, 5.12 / 100.0, True, 0.0, 1.0),
            (_hgbat, 5.0 / 100.0, True, -1.0, 1.0),
        ]

        return self._composition(X, xp, components, [20, 10, 10], [0, 200, 100])


class CEC2022F11(_CEC2022):
    """CEC 2022 F11: composition function 3 (expanded Schaffer's f6, modified
    Schwefel, Griewank, Rosenbrock, Rastrigin; f* = 2600).
    """

    _function_id = 11
    _bias = 2600.0

    def _function(self, X, xp):
        components = [
            (_expanded_schaffer_f6, 1.0, True, 0.0, 5.0e-4),
            (_modified_schwefel, 1000.0 / 100.0, True, 0.0, 1.0),
            (_griewank, 600.0 / 100.0, True, 0.0, 10.0),
            (_rosenbrock, 2.048 / 100.0, True, 1.0, 1.0),
            (_rastrigin, 5.12 / 100.0, True, 0.0, 10.0),
        ]

        return self._composition(
            X, xp, components, [20, 20, 30, 30, 20], [0, 200, 300, 400, 200]
        )


class CEC2022F12(_CEC2022):
    """CEC 2022 F12: composition function 4 (HGBat, Rastrigin, modified Schwefel,
    Bent Cigar, elliptic, expanded Schaffer's f6; f* = 2700).
    """

    _function_id = 12
    _bias = 2700.0

    def _function(self, X, xp):
        components = [
            (_hgbat, 5.0 / 100.0, True, -1.0, 10.0),
            (_rastrigin, 5.12 / 100.0, True, 0.0, 10.0),
            (_modified_schwefel, 1000.0 / 100.0, True, 0.0, 2.5),
            (_bent_cigar, 1.0, True, 0.0, 1.0e-26),
            (_elliptic, 1.0, True, 0.0, 1.0e-6),
            (_expanded_schaffer_f6, 1.0, True, 0.0, 5.0e-4),
        ]

        return self._composition(
            X,
            xp,
            components,
            [10, 20, 30, 40, 50, 60],
            [0, 300, 500, 100, 400, 200],
        )
