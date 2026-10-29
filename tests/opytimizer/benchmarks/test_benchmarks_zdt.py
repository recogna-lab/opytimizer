import numpy as np
import pytest

from opytimizer.benchmarks import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from opytimizer.benchmarks._utils import _non_dominated
from opytimizer.utils import exception as e

ZDT = [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]


def test_zdt_defaults():
    assert ZDT1().n_variables == 30
    assert ZDT4().n_variables == 10
    assert ZDT6().n_variables == 10
    assert ZDT4().lower_bound[:2] == [0.0, -5.0]

    with pytest.raises(e.ValueError):
        ZDT1(n_variables=1)


def test_zdt_known_values():
    x = np.zeros(30)

    assert np.allclose(ZDT1()(x), [0.0, 1.0])
    assert np.allclose(ZDT2()(x), [0.0, 1.0])
    assert np.allclose(ZDT1()(np.ones(30)), [1.0, 10.0 * (1.0 - np.sqrt(0.1))])


@pytest.mark.parametrize("cls", ZDT, ids=lambda c: c.__name__)
def test_zdt_pareto_set_maps_to_front(cls):
    p = cls()
    X = np.zeros((50, p.n_variables))
    X[:, 0] = np.linspace(0.0, 1.0, 50)
    F = p(X)

    front = p.pareto_front(5000)
    distance = np.min(np.linalg.norm(F[:, None, :] - front[None], axis=2), axis=1)

    # Every optimal solution is on (or, for ZDT3, dominated by) the front
    assert np.all(
        (distance < 1e-2) | np.array([np.any(np.all(front <= f, axis=1)) for f in F])
    )


@pytest.mark.parametrize("cls", ZDT, ids=lambda c: c.__name__)
def test_zdt_pareto_front_is_non_dominated(cls):
    front = cls().pareto_front(300)

    assert front.shape[1] == 2
    assert len(front) <= 300
    assert len(_non_dominated(front)) == len(front)
