import numpy as np
import pytest
from unittest.mock import MagicMock

import opytimizer.utils.exception as e
from opytimizer.optimizers.multi_objective.evolutionary.nsga2 import NSGA2


class DummyAgent:
    def __init__(self, fit=None):
        self.fit = np.array([0.0, 0.0]) if fit is None else np.array(fit)
        self.position = np.array([0.0])


@pytest.fixture
def opt():
    return NSGA2(crossover_operator=MagicMock(), mutation_operator=MagicMock())


def test_rank_property(opt):
    opt.rank = np.array([1, 2])
    assert np.array_equal(opt.rank, np.array([1, 2]))
    
    with pytest.raises(e.TypeError):
        opt.rank = [1, 2]


def test_crowding_distance_property(opt):
    opt.crowding_distance = np.array([1.5, 2.5])
    assert np.array_equal(opt.crowding_distance, np.array([1.5, 2.5]))
    
    with pytest.raises(e.TypeError):
        opt.crowding_distance = [1.5, 2.5]


def test_compile(opt):
    space = MagicMock()
    space.n_agents = 5
    
    opt.compile(space)
    
    assert len(opt.crowding_distance) == 5
    assert np.all(opt.crowding_distance == 0.0)


def test_fast_non_dominated_sort(opt):
    agents = [
        DummyAgent([1.0, 1.0]),
        DummyAgent([2.0, 2.0]),
        DummyAgent([0.5, 3.0])
    ]
    
    fronts, local_rank = opt._fast_non_dominated_sort(agents)
    
    assert len(fronts) > 0
    assert len(local_rank) == 3
    assert local_rank[0] == 0


def test_calculate_crowding_distance(opt):
    agents = [
        DummyAgent([1.0, 10.0]),
        DummyAgent([5.0, 5.0]),
        DummyAgent([10.0, 1.0])
    ]
    
    distances = opt._calculate_crowding_distance([0, 1, 2], agents)
    
    assert distances[0] == np.inf
    assert distances[2] == np.inf
    assert distances[1] > 0