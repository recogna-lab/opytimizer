"""CEC 2017 single-objective real-parameter benchmark suite (F1, F3-F30).

The functions follow the organizers' reference code (`cec17_test_func.cpp`),
which is what the published results of the competition were computed with,
and keep its numbering, which is also the numbering of the data files. F2
(shifted and rotated sum of different powers) is not provided: the final
definitions document excluded it for its unstable behaviour, and renumbered the
remaining functions as 1-29 (function k >= 3 here is function k - 1 there).

Every function is defined on [-100, 100]^D, keeps the official biases (F* =
100 k) and supports the dimensions with official data: 2, 10, 20, 30, 50 and
100 for F1-F10 and F21-F28; 10, 30, 50 and 100 for F11-F19, F29 and F30; and
10, 20, 30, 50 and 100 for F20. The data is not shipped with the package: it is
downloaded from the organizers' repository the first time a problem is created
and read when it is first evaluated.

References:
    N. H. Awad, M. Z. Ali, P. N. Suganthan, J. J. Liang, and B. Y. Qu, "Problem
    definitions and evaluation criteria for the CEC 2017 special session and
    competition on single objective real-parameter numerical optimization,"
    Technical Report, Nanyang Technological University, 2016.

"""

import functools
from typing import Optional

import numpy as np

from opytimizer.benchmarks._cec import (
    _CECBenchmark,
    _griewank_rosenbrock,
    _katsuura,
    _lunacek_bi_rastrigin,
    _modified_schwefel,
    _schaffer_f7,
    _segment,
)
from opytimizer.benchmarks._downloads import cec2017_data_dir
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
    _weierstrass,
    _zakharov,
)
from opytimizer.core.environment import Environment

# Number of components stored in the files of the composition functions
_CF_NUM = 10

_ALL_DIMENSIONS = (2, 10, 20, 30, 50, 100)
_HYBRID_DIMENSIONS = (10, 30, 50, 100)


@functools.lru_cache(maxsize=None)
def _load_data(function_id: int, n_variables: int):
    """Reads the shift vectors, rotation matrices and shuffles as the C code does.

    Functions below 20 read the first D shift entries and one D x D matrix;
    functions from 20 on read 10 rows of shifts (the first D entries of each) and
    10 matrices. F11-F20 read one shuffle and F29-F30 read 10 shuffles.

    Returns:
        (Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]): Shifts of shape
            (C, D), rotations of shape (C, D, D) and 0-based shuffles of shape
            (C, D) (None if unused).

    """

    n, directory = n_variables, cec2017_data_dir()

    # The C code reads F20 like a composition, but its files hold a single shift
    # and matrix, which are the only values the hybrid function uses
    components = 1 if function_id <= 20 else _CF_NUM

    # Some files hold fewer than 10 matrices (e.g., D = 2), more than any
    # function uses; the C code leaves the missing ones uninitialized
    values = np.loadtxt(directory / f"M_{function_id}_D{n}.txt").reshape(-1)
    components = min(components, values.size // (n * n))
    rotation = values[: components * n * n].reshape(components, n, n)

    path = directory / f"shift_data_{function_id}.txt"
    if function_id <= 20:
        shift = np.loadtxt(path).reshape(-1)[:n][None, :]
    else:
        shift = np.loadtxt(path, ndmin=2)[:_CF_NUM, :n]

    shuffle = None
    if 11 <= function_id <= 20 or function_id in (29, 30):
        count = 1 if function_id <= 20 else _CF_NUM
        path = directory / f"shuffle_data_{function_id}_D{n}.txt"
        indices = np.loadtxt(path, dtype=np.int64).reshape(-1)
        shuffle = indices[: count * n].reshape(count, n) - 1

    return shift, rotation, shuffle


def _bi_rastrigin_part(Y, start, stop, shift, xp):
    """Lunacek's bi-Rastrigin inside a hybrid (unrotated, mirrored by `shift`)."""
    return _lunacek_bi_rastrigin(Y[:, start:stop], shift[: stop - start], None, xp)


def _leading_schaffer_f7(Y, start, stop, shift, xp):
    """Schaffer's F7 inside a hybrid: the C code reads the first shuffled variables."""
    return _schaffer_f7(Y[:, : stop - start], xp)


# Hybrid functions as (fractions, parts); F29 and F30 compose some of them
_HF01 = ([0.2, 0.4, 0.4], [
    _segment(_zakharov),
    _segment(_rosenbrock, 2.048 / 100.0, 1.0),
    _segment(_rastrigin, 5.12 / 100.0),
])  # fmt: skip
_HF02 = ([0.3, 0.3, 0.4], [
    _segment(_elliptic),
    _segment(_modified_schwefel, 1000.0 / 100.0),
    _segment(_bent_cigar),
])  # fmt: skip
_HF03 = ([0.3, 0.3, 0.4], [
    _segment(_bent_cigar),
    _segment(_rosenbrock, 2.048 / 100.0, 1.0),
    _bi_rastrigin_part,
])  # fmt: skip
_HF04 = ([0.2, 0.2, 0.2, 0.4], [
    _segment(_elliptic),
    _segment(_ackley),
    _leading_schaffer_f7,
    _segment(_rastrigin, 5.12 / 100.0),
])  # fmt: skip
_HF05 = ([0.2, 0.2, 0.3, 0.3], [
    _segment(_bent_cigar),
    _segment(_hgbat, 5.0 / 100.0, -1.0),
    _segment(_rastrigin, 5.12 / 100.0),
    _segment(_rosenbrock, 2.048 / 100.0, 1.0),
])  # fmt: skip
_HF06 = ([0.2, 0.2, 0.3, 0.3], [
    _segment(_expanded_schaffer_f6),
    _segment(_hgbat, 5.0 / 100.0, -1.0),
    _segment(_rosenbrock, 2.048 / 100.0, 1.0),
    _segment(_modified_schwefel, 1000.0 / 100.0),
])  # fmt: skip
_HF07 = ([0.1, 0.2, 0.2, 0.2, 0.3], [
    _segment(_katsuura, 5.0 / 100.0),
    _segment(_ackley),
    _segment(_griewank_rosenbrock, 5.0 / 100.0),
    _segment(_modified_schwefel, 1000.0 / 100.0),
    _segment(_rastrigin, 5.12 / 100.0),
])  # fmt: skip
_HF08 = ([0.2, 0.2, 0.2, 0.2, 0.2], [
    _segment(_elliptic),
    _segment(_ackley),
    _segment(_rastrigin, 5.12 / 100.0),
    _segment(_hgbat, 5.0 / 100.0, -1.0),
    _segment(_discus),
])  # fmt: skip
_HF09 = ([0.2, 0.2, 0.2, 0.2, 0.2], [
    _segment(_bent_cigar),
    _segment(_rastrigin, 5.12 / 100.0),
    _segment(_griewank_rosenbrock, 5.0 / 100.0),
    _segment(_weierstrass, 0.5 / 100.0),
    _segment(_expanded_schaffer_f6),
])  # fmt: skip
_HF10 = ([0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [
    _segment(_hgbat, 5.0 / 100.0, -1.0),
    _segment(_katsuura, 5.0 / 100.0),
    _segment(_ackley),
    _segment(_rastrigin, 5.12 / 100.0),
    _segment(_modified_schwefel, 1000.0 / 100.0),
    _leading_schaffer_f7,
])  # fmt: skip


class _CEC2017(_CECBenchmark):
    """Base class of the CEC 2017 functions."""

    _dimensions = _ALL_DIMENSIONS

    def __init__(
        self, n_variables: int = 10, env: Optional[Environment] = None
    ) -> None:
        """Initialization method.

        The official data is downloaded on the first instantiation, if needed.

        Args:
            n_variables: Number of decision variables, one of the dimensions with
                official data for this function (see the module docstring).
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        super().__init__(n_variables, env)
        cec2017_data_dir()

    def _host_data(self):
        return _load_data(self._function_id, self.n_variables)


class _HybridCEC2017(_CEC2017):
    """Base class of the CEC 2017 hybrid functions (F11-F20)."""

    _dimensions = _HYBRID_DIMENSIONS
    _hybrid_definition = None

    def _function(self, X, xp):
        fractions, parts = self._hybrid_definition
        return self._hybrid(X, xp, fractions, parts)


class CEC2017F1(_CEC2017):
    """CEC 2017 F1: shifted and rotated Bent Cigar function (F* = 100)."""

    _function_id = 1
    _bias = 100.0

    def _function(self, X, xp):
        return _bent_cigar(self._shift_rotate(X, xp), xp)


class CEC2017F3(_CEC2017):
    """CEC 2017 F3: shifted and rotated Zakharov function (F* = 300).

    Function 2 of the final definitions document.
    """

    _function_id = 3
    _bias = 300.0

    def _function(self, X, xp):
        return _zakharov(self._shift_rotate(X, xp), xp)


class CEC2017F4(_CEC2017):
    """CEC 2017 F4: shifted and rotated Rosenbrock's function (F* = 400).

    Function 3 of the final definitions document.
    """

    _function_id = 4
    _bias = 400.0

    def _function(self, X, xp):
        return _rosenbrock(self._shift_rotate(X, xp, 2.048 / 100.0) + 1.0, xp)


class CEC2017F5(_CEC2017):
    """CEC 2017 F5: shifted and rotated Rastrigin's function (F* = 500).

    Function 4 of the final definitions document.
    """

    _function_id = 5
    _bias = 500.0

    def _function(self, X, xp):
        return _rastrigin(self._shift_rotate(X, xp, 5.12 / 100.0), xp)


class CEC2017F6(_CEC2017):
    """CEC 2017 F6: shifted Schaffer's F7 function (F* = 600).

    Function 5 of the final definitions document, which lists a rotated expanded
    Schaffer's F6; the official code evaluates Schaffer's F7 on the shifted
    variables, before the rotation.
    """

    _function_id = 6
    _bias = 600.0

    def _function(self, X, xp):
        return _schaffer_f7(self._shift_rotate(X, xp, rotate=False), xp)


class CEC2017F7(_CEC2017):
    """CEC 2017 F7: shifted and rotated Lunacek bi-Rastrigin function (F* = 700).

    Function 6 of the final definitions document.
    """

    _function_id = 7
    _bias = 700.0

    def _function(self, X, xp):
        shift, rotation, _ = self._data(xp)
        return _lunacek_bi_rastrigin(X - shift[0], shift[0], rotation[0], xp)


class CEC2017F8(_CEC2017):
    """CEC 2017 F8: shifted and rotated (non-continuous) Rastrigin's function.

    Function 7 of the final definitions document. As in the official code, the
    rounding step has no effect (F* = 800).
    """

    _function_id = 8
    _bias = 800.0

    def _function(self, X, xp):
        return _rastrigin(self._shift_rotate(X, xp, 5.12 / 100.0), xp)


class CEC2017F9(_CEC2017):
    """CEC 2017 F9: shifted and rotated Levy function (F* = 900).

    Function 8 of the final definitions document. As in the official code,
    w = 1 + (z - 1) / 4, so the minimum lies at x* = o + M^-1 1.
    """

    _function_id = 9
    _bias = 900.0

    def _function(self, X, xp):
        return _levy(self._shift_rotate(X, xp), xp)

    def _optimal_solution(self):
        shift, rotation, _ = self._host_data()
        n = self.n_variables

        # z = M (x - o) = 1  <=>  x = o + M^-1 1
        return shift[0] + np.linalg.solve(rotation[0], np.ones(n))


class CEC2017F10(_CEC2017):
    """CEC 2017 F10: shifted and rotated modified Schwefel's function (F* = 1000).

    Function 9 of the final definitions document.
    """

    _function_id = 10
    _bias = 1000.0

    def _function(self, X, xp):
        return _modified_schwefel(self._shift_rotate(X, xp, 1000.0 / 100.0), xp)


class CEC2017F11(_HybridCEC2017):
    """CEC 2017 F11: hybrid function 1 (Zakharov, Rosenbrock, Rastrigin; F* = 1100).

    Function 10 of the final definitions document.
    """

    _function_id = 11
    _bias = 1100.0
    _hybrid_definition = _HF01


class CEC2017F12(_HybridCEC2017):
    """CEC 2017 F12: hybrid function 2 (elliptic, modified Schwefel, Bent Cigar;
    F* = 1200).

    Function 11 of the final definitions document.
    """

    _function_id = 12
    _bias = 1200.0
    _hybrid_definition = _HF02


class CEC2017F13(_HybridCEC2017):
    """CEC 2017 F13: hybrid function 3 (Bent Cigar, Rosenbrock, Lunacek
    bi-Rastrigin; F* = 1300).

    Function 12 of the final definitions document.
    """

    _function_id = 13
    _bias = 1300.0
    _hybrid_definition = _HF03


class CEC2017F14(_HybridCEC2017):
    """CEC 2017 F14: hybrid function 4 (elliptic, Ackley, Schaffer's F7,
    Rastrigin; F* = 1400).

    Function 13 of the final definitions document.
    """

    _function_id = 14
    _bias = 1400.0
    _hybrid_definition = _HF04


class CEC2017F15(_HybridCEC2017):
    """CEC 2017 F15: hybrid function 5 (Bent Cigar, HGBat, Rastrigin,
    Rosenbrock; F* = 1500).

    Function 14 of the final definitions document.
    """

    _function_id = 15
    _bias = 1500.0
    _hybrid_definition = _HF05


class CEC2017F16(_HybridCEC2017):
    """CEC 2017 F16: hybrid function 6 (expanded Schaffer's F6, HGBat,
    Rosenbrock, modified Schwefel; F* = 1600).

    Function 15 of the final definitions document.
    """

    _function_id = 16
    _bias = 1600.0
    _hybrid_definition = _HF06


class CEC2017F17(_HybridCEC2017):
    """CEC 2017 F17: hybrid function 7 (Katsuura, Ackley, Griewank-Rosenbrock,
    modified Schwefel, Rastrigin; F* = 1700).

    Function 16 of the final definitions document.
    """

    _function_id = 17
    _bias = 1700.0
    _hybrid_definition = _HF07


class CEC2017F18(_HybridCEC2017):
    """CEC 2017 F18: hybrid function 8 (elliptic, Ackley, Rastrigin, HGBat,
    Discus; F* = 1800).

    Function 17 of the final definitions document.
    """

    _function_id = 18
    _bias = 1800.0
    _hybrid_definition = _HF08


class CEC2017F19(_HybridCEC2017):
    """CEC 2017 F19: hybrid function 9 (Bent Cigar, Rastrigin, Griewank-Rosenbrock,
    Weierstrass, expanded Schaffer's F6; F* = 1900).

    Function 18 of the final definitions document.
    """

    _function_id = 19
    _bias = 1900.0
    _hybrid_definition = _HF09


class CEC2017F20(_HybridCEC2017):
    """CEC 2017 F20: hybrid function 10 (HGBat, Katsuura, Ackley, Rastrigin,
    modified Schwefel, Schaffer's F7; F* = 2000).

    Function 19 of the final definitions document.
    """

    _function_id = 20
    _bias = 2000.0
    _dimensions = (10, 20, 30, 50, 100)
    _hybrid_definition = _HF10


class CEC2017F21(_CEC2017):
    """CEC 2017 F21: composition function 1 (Rosenbrock, elliptic, Rastrigin;
    F* = 2100).

    Function 20 of the final definitions document.
    """

    _function_id = 21
    _bias = 2100.0

    def _function(self, X, xp):
        components = [
            self._component(0, _rosenbrock, 2.048 / 100.0, offset=1.0),
            self._component(1, _elliptic, scale=1.0e-6),
            self._component(2, _rastrigin, 5.12 / 100.0),
        ]

        return self._composition(X, xp, components, [10, 20, 30], [0, 100, 200])


class CEC2017F22(_CEC2017):
    """CEC 2017 F22: composition function 2 (Rastrigin, Griewank, modified
    Schwefel; F* = 2200).

    Function 21 of the final definitions document.
    """

    _function_id = 22
    _bias = 2200.0

    def _function(self, X, xp):
        components = [
            self._component(0, _rastrigin, 5.12 / 100.0),
            self._component(1, _griewank, 600.0 / 100.0, scale=10.0),
            self._component(2, _modified_schwefel, 1000.0 / 100.0),
        ]

        return self._composition(X, xp, components, [10, 20, 30], [0, 100, 200])


class CEC2017F23(_CEC2017):
    """CEC 2017 F23: composition function 3 (Rosenbrock, Ackley, modified
    Schwefel, Rastrigin; F* = 2300).

    Function 22 of the final definitions document.
    """

    _function_id = 23
    _bias = 2300.0

    def _function(self, X, xp):
        components = [
            self._component(0, _rosenbrock, 2.048 / 100.0, offset=1.0),
            self._component(1, _ackley, scale=10.0),
            self._component(2, _modified_schwefel, 1000.0 / 100.0),
            self._component(3, _rastrigin, 5.12 / 100.0),
        ]

        return self._composition(
            X, xp, components, [10, 20, 30, 40], [0, 100, 200, 300]
        )


class CEC2017F24(_CEC2017):
    """CEC 2017 F24: composition function 4 (Ackley, elliptic, Griewank,
    Rastrigin; F* = 2400).

    Function 23 of the final definitions document.
    """

    _function_id = 24
    _bias = 2400.0

    def _function(self, X, xp):
        components = [
            self._component(0, _ackley, scale=10.0),
            self._component(1, _elliptic, scale=1.0e-6),
            self._component(2, _griewank, 600.0 / 100.0, scale=10.0),
            self._component(3, _rastrigin, 5.12 / 100.0),
        ]

        return self._composition(
            X, xp, components, [10, 20, 30, 40], [0, 100, 200, 300]
        )


class CEC2017F25(_CEC2017):
    """CEC 2017 F25: composition function 5 (Rastrigin, HappyCat, Ackley, Discus,
    Rosenbrock; F* = 2500).

    Function 24 of the final definitions document.
    """

    _function_id = 25
    _bias = 2500.0

    def _function(self, X, xp):
        components = [
            self._component(0, _rastrigin, 5.12 / 100.0, scale=10.0),
            self._component(1, _happycat, 5.0 / 100.0, offset=-1.0),
            self._component(2, _ackley, scale=10.0),
            self._component(3, _discus, scale=1.0e-6),
            self._component(4, _rosenbrock, 2.048 / 100.0, offset=1.0),
        ]

        return self._composition(
            X, xp, components, [10, 20, 30, 40, 50], [0, 100, 200, 300, 400]
        )


class CEC2017F26(_CEC2017):
    """CEC 2017 F26: composition function 6 (expanded Schaffer's F6, modified
    Schwefel, Griewank, Rosenbrock, Rastrigin; F* = 2600).

    Function 25 of the final definitions document.
    """

    _function_id = 26
    _bias = 2600.0

    def _function(self, X, xp):
        components = [
            self._component(0, _expanded_schaffer_f6, scale=5.0e-4),
            self._component(1, _modified_schwefel, 1000.0 / 100.0),
            self._component(2, _griewank, 600.0 / 100.0, scale=10.0),
            self._component(3, _rosenbrock, 2.048 / 100.0, offset=1.0),
            self._component(4, _rastrigin, 5.12 / 100.0, scale=10.0),
        ]

        return self._composition(
            X, xp, components, [10, 20, 20, 30, 40], [0, 100, 200, 300, 400]
        )


class CEC2017F27(_CEC2017):
    """CEC 2017 F27: composition function 7 (HGBat, Rastrigin, modified Schwefel,
    Bent Cigar, elliptic, expanded Schaffer's F6; F* = 2700).

    Function 26 of the final definitions document.
    """

    _function_id = 27
    _bias = 2700.0

    def _function(self, X, xp):
        components = [
            self._component(0, _hgbat, 5.0 / 100.0, offset=-1.0, scale=10.0),
            self._component(1, _rastrigin, 5.12 / 100.0, scale=10.0),
            self._component(2, _modified_schwefel, 1000.0 / 100.0, scale=2.5),
            self._component(3, _bent_cigar, scale=1.0e-26),
            self._component(4, _elliptic, scale=1.0e-6),
            self._component(5, _expanded_schaffer_f6, scale=5.0e-4),
        ]

        return self._composition(
            X,
            xp,
            components,
            [10, 20, 30, 40, 50, 60],
            [0, 100, 200, 300, 400, 500],
        )


class CEC2017F28(_CEC2017):
    """CEC 2017 F28: composition function 8 (Ackley, Griewank, Discus, Rosenbrock,
    HappyCat, expanded Schaffer's F6; F* = 2800).

    Function 27 of the final definitions document.
    """

    _function_id = 28
    _bias = 2800.0

    def _function(self, X, xp):
        components = [
            self._component(0, _ackley, scale=10.0),
            self._component(1, _griewank, 600.0 / 100.0, scale=10.0),
            self._component(2, _discus, scale=1.0e-6),
            self._component(3, _rosenbrock, 2.048 / 100.0, offset=1.0),
            self._component(4, _happycat, 5.0 / 100.0, offset=-1.0),
            self._component(5, _expanded_schaffer_f6, scale=5.0e-4),
        ]

        return self._composition(
            X,
            xp,
            components,
            [10, 20, 30, 40, 50, 60],
            [0, 100, 200, 300, 400, 500],
        )


class _HybridCompositionCEC2017(_CEC2017):
    """Base class of the compositions of hybrid functions (F29 and F30)."""

    _dimensions = _HYBRID_DIMENSIONS
    _hybrids = ()

    def _function(self, X, xp):
        # Each component is a hybrid function with its own shift, rotation and shuffle
        components = [
            functools.partial(self._hybrid, fractions=f, parts=p, component=i)
            for i, (f, p) in enumerate(self._hybrids)
        ]

        return self._composition(X, xp, components, [10, 30, 50], [0, 100, 200])


class CEC2017F29(_HybridCompositionCEC2017):
    """CEC 2017 F29: composition function 9 (hybrid functions 5, 6 and 7;
    F* = 2900).

    Function 28 of the final definitions document.
    """

    _function_id = 29
    _bias = 2900.0
    _hybrids = (_HF05, _HF06, _HF07)


class CEC2017F30(_HybridCompositionCEC2017):
    """CEC 2017 F30: composition function 10 (hybrid functions 5, 8 and 9;
    F* = 3000).

    Function 29 of the final definitions document.
    """

    _function_id = 30
    _bias = 3000.0
    _hybrids = (_HF05, _HF08, _HF09)
