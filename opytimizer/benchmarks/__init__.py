"""Benchmark problems from the literature, ready to be used with Opytimizer.

Each problem chooses its number of variables (and objectives, when scalable)
and the `Environment` (numpy / cupy) that runs it. Problems are callables, so
they can be wrapped by `Function` and work with both standard and tensorized
spaces:

    >>> from opytimizer.benchmarks import DTLZ2
    >>> problem = DTLZ2(n_variables=12, n_objectives=3)
    >>> space = SearchSpace(100, problem.n_variables, problem.n_objectives,
    ...                     problem.lower_bound, problem.upper_bound)
    >>> function = Function(problem)
    >>> reference = problem.pareto_front(n_points=500)

CuPy is only imported when a CuPy environment is actually used.
"""

from opytimizer.benchmarks.cec2017 import (
    CEC2017F1,
    CEC2017F3,
    CEC2017F4,
    CEC2017F5,
    CEC2017F6,
    CEC2017F7,
    CEC2017F8,
    CEC2017F9,
    CEC2017F10,
    CEC2017F11,
    CEC2017F12,
    CEC2017F13,
    CEC2017F14,
    CEC2017F15,
    CEC2017F16,
    CEC2017F17,
    CEC2017F18,
    CEC2017F19,
    CEC2017F20,
    CEC2017F21,
    CEC2017F22,
    CEC2017F23,
    CEC2017F24,
    CEC2017F25,
    CEC2017F26,
    CEC2017F27,
    CEC2017F28,
    CEC2017F29,
    CEC2017F30,
)
from opytimizer.benchmarks.cec2022 import (
    CEC2022F1,
    CEC2022F2,
    CEC2022F3,
    CEC2022F4,
    CEC2022F5,
    CEC2022F6,
    CEC2022F7,
    CEC2022F8,
    CEC2022F9,
    CEC2022F10,
    CEC2022F11,
    CEC2022F12,
)
from opytimizer.benchmarks.multi_objective import *  # noqa: F401,F403
from opytimizer.benchmarks.multi_objective import __all__ as _multi_objective_all
from opytimizer.benchmarks.single_objective import (
    Ackley,
    Alpine,
    BentCigar,
    Discus,
    DixonPrice,
    ExpandedSchafferF6,
    Griewank,
    HappyCat,
    HGBat,
    HighConditionedElliptic,
    Levy,
    Quartic,
    Rastrigin,
    Rosenbrock,
    Salomon,
    Schwefel,
    SchwefelProblem12,
    SchwefelProblem222,
    Sphere,
    Step,
    StyblinskiTang,
    SumSquares,
    Weierstrass,
    Zakharov,
)

__all__ = [
    "Ackley", "Alpine", "BentCigar", "Discus", "DixonPrice", "ExpandedSchafferF6",
    "Griewank", "HappyCat", "HGBat", "HighConditionedElliptic", "Levy", "Quartic",
    "Rastrigin", "Rosenbrock", "Salomon", "Schwefel", "SchwefelProblem12",
    "SchwefelProblem222", "Sphere", "Step", "StyblinskiTang", "SumSquares",
    "Weierstrass", "Zakharov",
    "CEC2017F1", "CEC2017F3", "CEC2017F4", "CEC2017F5", "CEC2017F6", "CEC2017F7",
    "CEC2017F8", "CEC2017F9", "CEC2017F10", "CEC2017F11", "CEC2017F12", "CEC2017F13",
    "CEC2017F14", "CEC2017F15", "CEC2017F16", "CEC2017F17", "CEC2017F18", "CEC2017F19",
    "CEC2017F20", "CEC2017F21", "CEC2017F22", "CEC2017F23", "CEC2017F24", "CEC2017F25",
    "CEC2017F26", "CEC2017F27", "CEC2017F28", "CEC2017F29", "CEC2017F30",
    "CEC2022F1", "CEC2022F2", "CEC2022F3", "CEC2022F4", "CEC2022F5", "CEC2022F6",
    "CEC2022F7", "CEC2022F8", "CEC2022F9", "CEC2022F10", "CEC2022F11", "CEC2022F12",
] + list(_multi_objective_all)  # fmt: skip
