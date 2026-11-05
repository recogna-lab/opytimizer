"""CEC 2022 single-objective bound-constrained benchmark suite (F1-F12).

The functions follow the organizers' reference C code (`cec22_test_func.cpp`),
which is what the published results of the competition were computed with.

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
from pathlib import Path

import numpy as np

from opytimizer.benchmarks._cec import (
    _CECBenchmark,
    _griewank_rosenbrock,
    _katsuura,
    _modified_schwefel,
    _schaffer_f7,
    _segment,
)
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

_DATA_DIR = Path(__file__).resolve().parent / "data" / "cec2022"


@functools.lru_cache(maxsize=None)
def _load_data(function_id: int, n_variables: int):
    """Reads the shift vectors, rotation matrices and shuffle of a function.

    Returns:
        (Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]): Shifts of shape
            (C, D), rotations of shape (C, D, D) and the 0-based shuffle of shape
            (1, D) (only for the hybrid functions), C being the number of components.

    """

    n = n_variables
    shift = np.loadtxt(_DATA_DIR / f"shift_data_{function_id}.txt", ndmin=2)[:, :n]
    rotation = np.loadtxt(_DATA_DIR / f"M_{function_id}_D{n}.txt").reshape(-1, n, n)

    shuffle = None
    if 6 <= function_id <= 8:
        path = _DATA_DIR / f"shuffle_data_{function_id}_D{n}.txt"
        shuffle = np.loadtxt(path, dtype=np.int64).reshape(1, -1)[:, :n] - 1

    return shift, rotation, shuffle


class _CEC2022(_CECBenchmark):
    """Base class of the CEC 2022 functions."""

    _dimensions = (2, 10, 20)

    def _host_data(self):
        return _load_data(self._function_id, self.n_variables)


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
            lambda Y, start, stop, shift, xp: _schaffer_f7(Y[:, : stop - start], xp),
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
            self._component(0, _rosenbrock, 2.048 / 100.0, True, 1.0, 1.0),
            self._component(1, _elliptic, 1.0, True, 0.0, 1.0e-6),
            self._component(2, _bent_cigar, 1.0, True, 0.0, 1.0e-26),
            self._component(3, _discus, 1.0, True, 0.0, 1.0e-6),
            self._component(4, _elliptic, 1.0, False, 0.0, 1.0e-6),
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
            self._component(0, _modified_schwefel, 1000.0 / 100.0, False, 0.0, 1.0),
            self._component(1, _rastrigin, 5.12 / 100.0, True, 0.0, 1.0),
            self._component(2, _hgbat, 5.0 / 100.0, True, -1.0, 1.0),
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
            self._component(0, _expanded_schaffer_f6, 1.0, True, 0.0, 5.0e-4),
            self._component(1, _modified_schwefel, 1000.0 / 100.0, True, 0.0, 1.0),
            self._component(2, _griewank, 600.0 / 100.0, True, 0.0, 10.0),
            self._component(3, _rosenbrock, 2.048 / 100.0, True, 1.0, 1.0),
            self._component(4, _rastrigin, 5.12 / 100.0, True, 0.0, 10.0),
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
            self._component(0, _hgbat, 5.0 / 100.0, True, -1.0, 10.0),
            self._component(1, _rastrigin, 5.12 / 100.0, True, 0.0, 10.0),
            self._component(2, _modified_schwefel, 1000.0 / 100.0, True, 0.0, 2.5),
            self._component(3, _bent_cigar, 1.0, True, 0.0, 1.0e-26),
            self._component(4, _elliptic, 1.0, True, 0.0, 1.0e-6),
            self._component(5, _expanded_schaffer_f6, 1.0, True, 0.0, 5.0e-4),
        ]

        return self._composition(
            X,
            xp,
            components,
            [10, 20, 30, 40, 50, 60],
            [0, 300, 500, 100, 400, 200],
        )
