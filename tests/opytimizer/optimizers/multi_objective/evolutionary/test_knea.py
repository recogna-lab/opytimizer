import numpy as np
import pytest

from opytimizer.core.agent import Agent
from opytimizer.spaces.search import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary.knea import KnEA
import opytimizer.utils.exception as e


def test_knea_properties():
    # Test property setters and validation constraints
    new_knea = KnEA(k=3, T=0.5)
    
    assert new_knea.knn_num == 3
    assert new_knea.T == 0.5

    with pytest.raises(e.TypeError):
        new_knea.knn_num = "3"
    with pytest.raises(e.ValueError):
        new_knea.knn_num = 0

    with pytest.raises(e.TypeError):
        new_knea.T = "0.5"
    with pytest.raises(e.ValueError):
        new_knea.T = 1.5


def test_knea_compile():
    search_space = SearchSpace(
        n_agents=5,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_knea = KnEA()
    new_knea.compile(search_space)

    # Ensure state arrays are initialized with twice the agent count (2N)
    assert len(new_knea.r) == 10
    assert len(new_knea.t) == 10
    assert np.all(new_knea.r == -1)
    assert np.all(new_knea.t == -1)
    assert isinstance(new_knea.K, list)
    assert len(new_knea.K) == 0


def make_agent(fit):
    """Helper to instantiate mock agents with specific fitness values."""
    a = Agent(n_variables=3, n_dimensions=1, n_objectives=2, 
              lower_bound=[0, 0, 0], upper_bound=[1, 1, 1])
    a.fit = np.array(fit)
    a.position = np.random.rand(3, 1)  # Required for genetic operator evaluation
    return a


def test_knea_precompute_weighted_distances():
    agents = [
        make_agent([1.0, 2.0]),
        make_agent([2.0, 1.0]),
        make_agent([1.5, 1.5]),
        make_agent([3.0, 3.0]),
        make_agent([0.5, 2.5]),
    ]
    
    new_knea = KnEA(k=2)
    new_knea._precompute_weighted_distances(agents)
    
    assert new_knea._weighted_dists is not None
    assert len(new_knea._weighted_dists) == len(agents)
    assert isinstance(new_knea._weighted_dists, np.ndarray)


def test_knea_mating_selection():
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5]),
        make_agent([3, 3]),
        make_agent([0.5, 2.5]),
    ]
    
    new_knea = KnEA(k=2)
    new_knea._precompute_weighted_distances(agents)
    
    # Mock knee points set
    K_mock = [agents[0], agents[1]]
    N_to_select = 3
    
    mating_pool = new_knea._mating_selection(agents, K_mock, N_to_select)
    
    assert isinstance(mating_pool, list)
    assert len(mating_pool) == N_to_select
    assert all(isinstance(a, Agent) for a in mating_pool)


def test_knea_genetic_operators():
    def dummy_func(x):
        return np.array([np.sum(x), np.sum(x**2)])
        
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5]),
        make_agent([3, 3])
    ]
    
    new_knea = KnEA()
    
    # N must be even for parent pairing
    offsprings = new_knea._genetic_operators(mating=agents, N=len(agents), function=dummy_func)
    
    assert isinstance(offsprings, list)
    assert len(offsprings) == len(agents)
    assert type(offsprings[0]).__name__ == "Agent"


def test_knea_fast_non_dominated_sort():
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5]),
        make_agent([3, 3]),
        make_agent([0.5, 2.5]),
    ]
    new_knea = KnEA()
    fronts = new_knea._fast_non_dominated_sort(agents)
    
    assert isinstance(fronts, list)
    assert len(fronts) > 0
    
    # Agent [3, 3] at index 3 is dominated and should not be in the first front
    assert 3 not in fronts[0]


def test_knea_finding_knee_point():
    agents = [
        make_agent([1, 10]),
        make_agent([2, 8]),
        make_agent([5, 5]),
        make_agent([8, 2]),
        make_agent([10, 1]),
    ]
    new_knea = KnEA()
    
    # Mock pre-compiled state variables
    new_knea.r = -np.ones(10)
    new_knea.t = -np.ones(10)
    
    fronts = new_knea._fast_non_dominated_sort(agents)
    knee_indices, sorted_fronts, front_map = new_knea._finding_knee_point(agents, fronts)
    
    assert isinstance(knee_indices, list)
    assert isinstance(sorted_fronts, list)
    assert isinstance(front_map, list)
    assert len(knee_indices) == len(sorted_fronts)


def test_knea_environmental_selection():
    agents = [
        make_agent([1, 10]),
        make_agent([2, 8]),
        make_agent([5, 5]),
        make_agent([8, 2]),
        make_agent([10, 1]),
    ]
    new_knea = KnEA()
    
    new_knea.r = -np.ones(10)
    new_knea.t = -np.ones(10)
    
    fronts = new_knea._fast_non_dominated_sort(agents)
    knee_indices, sorted_fronts, front_map = new_knea._finding_knee_point(agents, fronts)
    
    # Request exactly 3 agents to trigger truncation logic
    survivors = new_knea._environmental_selection(
        agents, fronts, knee_indices, sorted_fronts, front_map, N=3
    )
    
    assert isinstance(survivors, list)
    assert len(survivors) == 3


def test_knea_evaluate():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=4,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_knea = KnEA()
    new_knea.compile(search_space)
    new_knea.evaluate(search_space, multi_square)

    assert search_space.agents[0].fit is not None
    assert not new_knea.is_first_generation


def test_knea_update():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=4,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    # knn_num must be strictly less than population size
    new_knea = KnEA(k=2)
    new_knea.compile(search_space)
    
    new_knea.evaluate(search_space, multi_square)
    new_knea.update(search_space, multi_square)

    assert len(search_space.agents) == 4
    assert isinstance(new_knea.K, list)