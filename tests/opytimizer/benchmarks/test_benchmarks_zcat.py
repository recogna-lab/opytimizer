import itertools

import numpy as np
import pytest

from opytimizer import benchmarks
from opytimizer.benchmarks._utils import _non_dominated
from opytimizer.benchmarks.multi_objective.zcat import _COMPLICATED_G, _g, _shape
from opytimizer.utils import exception as e

ZCAT = [getattr(benchmarks, f"ZCAT{i}") for i in range(1, 21)]


def test_zcat_defaults():
    p = benchmarks.ZCAT1()

    assert (p.n_variables, p.n_objectives, p.level) == (30, 2, 1)
    assert p.lower_bound[:2] == [-0.5, -1.0]
    assert p.upper_bound[:2] == [0.5, 1.0]


def test_zcat_invalid_arguments():
    with pytest.raises(e.ValueError):
        benchmarks.ZCAT1(level=7)

    with pytest.raises(e.ValueError):
        benchmarks.ZCAT1(n_variables=2, n_objectives=5)

    # One-dimensional Pareto sets only need a single variable
    assert benchmarks.ZCAT14(n_variables=1, n_objectives=5).n_variables == 1


@pytest.mark.parametrize(
    "cls", [c for c in ZCAT if c._problem_id not in (19, 20)], ids=lambda c: c.__name__
)
@pytest.mark.parametrize("complicated", [False, True])
def test_zcat_pareto_set_has_no_distance(cls, complicated):
    m, n = 3, 12
    # Level 1 and no bias: sqrt / |z|^0.05 would amplify the normalization round-off
    p = cls(n, m, complicated_pareto_set=complicated)
    d = 1 if p._problem_id in (14, 15, 16) else m - 1

    # Optimal solutions: y_II = g(y_I)
    yI = np.random.default_rng(0).random((60, d))
    g_id = _COMPLICATED_G[p._problem_id] if complicated else 0
    y = np.concatenate((yI, _g(g_id, yI, n, np)), axis=1)
    lb, ub = np.array(p.lower_bound), np.array(p.upper_bound)

    F = p(lb + y * (ub - lb))
    expected = np.arange(1, m + 1) ** 2 * _shape(p._problem_id, y, m, 1e-12, np)

    assert np.allclose(F, expected, atol=1e-9)


@pytest.mark.parametrize("cls", ZCAT, ids=lambda c: c.__name__)
@pytest.mark.parametrize("m", [2, 3])
def test_zcat_pareto_front(cls, m):
    front = cls(n_objectives=m).pareto_front(300)

    assert front.shape[1] == m
    assert 0 < len(front) <= 300
    assert len(_non_dominated(front)) == len(front)


@pytest.mark.parametrize("cls", ZCAT, ids=lambda c: c.__name__)
def test_zcat_features_are_finite(cls):
    X = np.random.default_rng(1).random((20, 10)) - 0.5
    for level, bias, imbalance, complicated in itertools.product(
        range(1, 7), [False, True], [False, True], [False, True]
    ):
        p = cls(
            10,
            3,
            level=level,
            bias=bias,
            imbalance=imbalance,
            complicated_pareto_set=complicated,
        )

        assert np.all(np.isfinite(p(X * np.array(p.upper_bound) * 2)))
