import numpy as np
import pytest

from opytimizer.benchmarks import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from opytimizer.benchmarks._utils import _non_dominated
from opytimizer.utils import exception as e

DTLZ = [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]


def _pareto_set(p, n=100, seed=0):
    """Random optimal solutions: free position variables, optimal distance ones."""
    m = p.n_objectives
    X = np.random.default_rng(seed).random((n, p.n_variables))
    X[:, m - 1 :] = 0.0 if isinstance(p, (DTLZ6, DTLZ7)) else 0.5
    return X


def test_dtlz_defaults():
    assert DTLZ1().n_variables == 7
    assert DTLZ2().n_variables == 12
    assert DTLZ7().n_variables == 22
    assert DTLZ2(n_objectives=5).n_variables == 14

    with pytest.raises(e.ValueError):
        DTLZ2(n_variables=2, n_objectives=3)

    with pytest.raises(e.ValueError):
        DTLZ2(n_objectives=1)


@pytest.mark.parametrize("m", [2, 3, 5, 10])
def test_dtlz_linear_and_spherical_fronts(m):
    F1 = DTLZ1(n_objectives=m)(_pareto_set(DTLZ1(n_objectives=m)))
    assert np.allclose(F1.sum(axis=1), 0.5)

    for cls in (DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6):
        p = cls(n_objectives=m)
        assert np.allclose(np.linalg.norm(p(_pareto_set(p)), axis=1), 1.0)

    assert np.allclose(DTLZ1(n_objectives=m).pareto_front(200).sum(axis=1), 0.5)
    assert np.allclose(np.linalg.norm(DTLZ2(n_objectives=m).pareto_front(200), axis=1), 1.0)


@pytest.mark.parametrize("cls", DTLZ, ids=lambda c: c.__name__)
@pytest.mark.parametrize("m", [2, 3, 4])
def test_dtlz_pareto_front(cls, m):
    front = cls(n_objectives=m).pareto_front(300)

    assert front.shape[1] == m
    assert 0 < len(front) <= 300
    assert len(_non_dominated(front)) == len(front)


def test_dtlz7_pareto_set_on_front():
    p = DTLZ7(n_objectives=3)
    front = p.pareto_front(2000)
    F = _non_dominated(p(_pareto_set(p, 500)))

    distance = np.min(np.linalg.norm(F[:, None, :] - front[None], axis=2), axis=1)
    assert np.median(distance) < 0.05
