import numpy as np
import pytest

from opytimizer.benchmarks import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from opytimizer.benchmarks._utils import _non_dominated
from opytimizer.utils import exception as e

WFG = [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]


def _pareto_set(p, n=100, seed=0):
    """Optimal solutions of WFG1-7: z_i = 0.35 * 2i for every distance variable."""
    upper = np.array(p.upper_bound)
    Z = np.random.default_rng(seed).random((n, p.n_variables)) * upper
    Z[:, p.k :] = 0.35 * upper[p.k :]
    return Z


def test_wfg_defaults():
    p = WFG4()

    assert (p.n_objectives, p.k, p.l, p.n_variables) == (3, 4, 20, 24)
    assert WFG4(n_objectives=2).k == 4
    assert p.upper_bound[:3] == [2.0, 4.0, 6.0]


def test_wfg_invalid_arguments():
    with pytest.raises(e.ValueError):
        WFG4(n_objectives=3, k=3)

    with pytest.raises(e.ValueError):
        WFG4(n_variables=4, n_objectives=3, k=4)

    with pytest.raises(e.ValueError):
        WFG2(n_variables=9, n_objectives=3, k=4)


@pytest.mark.parametrize("cls", [WFG4, WFG5, WFG6, WFG7], ids=lambda c: c.__name__)
@pytest.mark.parametrize("m", [2, 3, 5])
def test_wfg_concave_pareto_set(cls, m):
    p = cls(n_objectives=m)
    F = p(_pareto_set(p))
    scale = 2.0 * np.arange(1, m + 1)

    assert np.allclose(np.linalg.norm(F / scale, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(p.pareto_front(200) / scale, axis=1), 1.0)


@pytest.mark.parametrize("m", [2, 3])
def test_wfg3_linear_degenerate_front(m):
    p = WFG3(n_objectives=m)
    scale = 2.0 * np.arange(1, m + 1)
    front = p.pareto_front(100) / scale

    if m == 2:
        assert np.allclose(front.sum(axis=1), 1.0)
    assert np.allclose(
        np.min(
            np.abs(p(_pareto_set(p))[:, None] / scale - front[None]).sum(axis=2), axis=1
        ),
        0.0,
        atol=0.05,
    )


@pytest.mark.parametrize("cls", WFG, ids=lambda c: c.__name__)
@pytest.mark.parametrize("m", [2, 3])
def test_wfg_pareto_front(cls, m):
    p = cls(n_objectives=m)
    front = p.pareto_front(300)

    assert front.shape[1] == m
    assert 0 < len(front) <= 300
    assert len(_non_dominated(front)) == len(front)
    assert np.all(
        np.isfinite(
            p(
                np.random.default_rng(1).random((50, p.n_variables))
                * np.array(p.upper_bound)
            )
        )
    )
