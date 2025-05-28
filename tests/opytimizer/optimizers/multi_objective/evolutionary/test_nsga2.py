import numpy as np
from opytimizer.optimizers.multi_objective.evolutionary import nsga2
from opytimizer.spaces.search import SearchSpace
from opytimizer.core.agent import Agent


def test_nsga2_params():
    params = {
        "crossover_rate": 0.9,
        "mutation_rate": 0.025
    }

    new_nsga2 = nsga2.NSGA2(params=params)
    
    assert new_nsga2.crossover_rate == 0.9
    
    assert new_nsga2.mutation_rate == 0.025


def test_nsga2_params_setter():
    new_nsga2 = nsga2.NSGA2()

    try:
        new_nsga2.crossover_rate = "a"
    except:
        new_nsga2.crossover_rate = 0.9

    try:
        new_nsga2.crossover_rate = -1
    except:
        new_nsga2.crossover_rate = 0.9

    assert new_nsga2.crossover_rate == 0.9

    try:
        new_nsga2.mutation_rate = "b"
    except:
        new_nsga2.mutation_rate = 0.025

    try:
        new_nsga2.mutation_rate = -1
    except:
        new_nsga2.mutation_rate = 0.025

    assert new_nsga2.mutation_rate == 0.025


def test_nsga2_compile():
    search_space = SearchSpace(
        n_agents=2, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_nsga2 = nsga2.NSGA2()
    new_nsga2.compile(search_space)
    
    try:
        new_nsga2.rank = 1
    except:
        new_nsga2.rank = np.array([1])

    assert new_nsga2.rank == np.array([1])

    try:
        new_nsga2.crowding_distance = 1
    except:
        new_nsga2.crowding_distance = np.array([1])

    assert new_nsga2.crowding_distance == np.array([1])

def test_nsga2_crossover():
    search_space = SearchSpace(
        n_agents=10, n_variables=2, n_objectives=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_nsga2 = nsga2.NSGA2()

    alpha, beta = new_nsga2._crossover(search_space.agents[0], search_space.agents[1])

    assert type(alpha).__name__ == "Agent"
    assert type(beta).__name__ == "Agent"


def test_nsga2_mutation():
    search_space = SearchSpace(
        n_agents=10, n_variables=2, n_objectives=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_nsga2 = nsga2.NSGA2()

    alpha = new_nsga2._mutation(search_space.agents[0])
    beta = new_nsga2._mutation(search_space.agents[1])

    assert type(alpha).__name__ == "Agent"
    assert type(beta).__name__ == "Agent"

def make_agent(fit):
    a = Agent(3, 1, 2, [0, 0, 0], [1, 1, 1])
    a.fit = np.array(fit)
    return a

def test_nsga2_fast_non_dominated_sort():
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5]),
        make_agent([3, 3]),
        make_agent([0.5, 2.5])
    ]
    new_nsga2 = nsga2.NSGA2()
    fronts = new_nsga2._fast_non_dominated_sort(agents)
    assert isinstance(fronts, list)
    assert len(fronts) > 0
    assert 0 in fronts[0] or 1 in fronts[0]


def test_nsga2_crowding_distance():
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5])
    ]
    new_nsga2 = nsga2.NSGA2()
    front = [0, 1, 2]
    distances = new_nsga2._calculate_crowding_distance(front, agents)
    assert isinstance(distances, np.ndarray)
    assert len(distances) == 3

def test_nsga2_tournament_selection():
    new_nsga2 = nsga2.NSGA2()
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5]),
        make_agent([3, 3]),
        make_agent([0.5, 2.5])
    ]

    new_nsga2._fast_non_dominated_sort(agents)
    
    front = [0, 1, 2, 3, 4]
    
    new_nsga2.crowding_distance = new_nsga2._calculate_crowding_distance(front, agents)

    selected = new_nsga2._tournament_selection(agents)
    
    assert isinstance(selected, list) or isinstance(selected, Agent) or isinstance(selected, int)
    assert len(selected) == 5

def test_nsga2_evaluate():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=2, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_nsga2 = nsga2.NSGA2()
    new_nsga2.compile(search_space)
    
    new_nsga2.evaluate(search_space, multi_square)
    
    assert isinstance(search_space.pareto_front, list)
    assert len(search_space.pareto_front) > 0


def test_nsga2_update():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])
    
    search_space = SearchSpace(
        n_agents=2, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_nsga2 = nsga2.NSGA2()
    new_nsga2.compile(search_space)
    
    new_nsga2.update(search_space, multi_square)