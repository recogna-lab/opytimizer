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
    "CEC2022F1", "CEC2022F2", "CEC2022F3", "CEC2022F4", "CEC2022F5", "CEC2022F6",
    "CEC2022F7", "CEC2022F8", "CEC2022F9", "CEC2022F10", "CEC2022F11", "CEC2022F12",
] + list(_multi_objective_all)  # fmt: skip
