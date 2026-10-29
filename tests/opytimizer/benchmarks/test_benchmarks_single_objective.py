import numpy as np
import pytest

from opytimizer import benchmarks
from opytimizer.benchmarks import single_objective

SINGLE_OBJECTIVE = [
    getattr(benchmarks, name)
    for name in benchmarks.__all__
    if hasattr(single_objective, name)
]


@pytest.mark.parametrize("cls", SINGLE_OBJECTIVE, ids=lambda c: c.__name__)
@pytest.mark.parametrize("n_variables", [2, 10, 30])
def test_single_objective_optimum(cls, n_variables):
    p = cls(n_variables=n_variables)
    x = p.optimal_solution()

    assert x.shape == (n_variables,)
    assert np.isclose(float(p(x)), p.optimal_value, atol=1e-6)


@pytest.mark.parametrize("cls", SINGLE_OBJECTIVE, ids=lambda c: c.__name__)
def test_single_objective_optimum_is_minimal(cls):
    p = cls(n_variables=5)
    rng = np.random.default_rng(0)

    lb, ub = np.array(p.lower_bound), np.array(p.upper_bound)
    X = lb + rng.random((1000, 5)) * (ub - lb)

    assert np.all(p(X) >= p.optimal_value - 1e-9)


@pytest.mark.parametrize("cls", SINGLE_OBJECTIVE, ids=lambda c: c.__name__)
def test_single_objective_bounds(cls):
    p = cls(n_variables=3)

    assert len(p.lower_bound) == len(p.upper_bound) == 3
    assert all(lb < ub for lb, ub in zip(p.lower_bound, p.upper_bound))


def test_single_objective_known_values():
    x = np.ones(2)

    assert np.isclose(benchmarks.Sphere(2)(x), 2.0)
    assert np.isclose(benchmarks.SumSquares(2)(x), 3.0)
    assert np.isclose(benchmarks.Rastrigin(2)(x), 2.0)
    assert np.isclose(benchmarks.Rosenbrock(2)(np.zeros(2)), 1.0)
    assert np.isclose(benchmarks.Zakharov(2)(x), 2.0 + 1.5**2 + 1.5**4)
