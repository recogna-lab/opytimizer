import numpy as np

from opytimizer.optimizers.multi_objective.evolutionary import moead
from opytimizer.spaces.search import SearchSpace
from opytimizer.utils.weights_vector import das_dennis


def test_moead_params():
    params = {"n_subproblems": 100, "neighborhood_size": 10, "pbi_penalty": 6}

    new_moead = moead.MOEAD(params=params)

    assert new_moead.n_subproblems == 100
    assert new_moead.neighborhood_size == 10
    assert isinstance(new_moead.pbi_penalty, float)
    assert new_moead.pbi_penalty == 6.0


def test_moead_params_setter():
    new_moead = moead.MOEAD()

    try:
        new_moead.n_subproblems = "c"
    except:
        new_moead.n_subproblems = 100

    try:
        new_moead.n_subproblems = 0
    except:
        new_moead.n_subproblems = 100

    assert new_moead.n_subproblems == 100

    try:
        new_moead.neighborhood_size = "d"
    except:
        new_moead.neighborhood_size = 10

    try:
        new_moead.neighborhood_size = 1
    except:
        new_moead.neighborhood_size = 10

    assert new_moead.neighborhood_size == 10
    
    try:
        new_moead.pbi_penalty = 'a'
    except:
        new_moead.pbi_penalty = 5
        
    assert new_moead.pbi_penalty == 5.0


def test_moead_compile():
    weights, n_agents = das_dennis(2, 2)
    search_space = SearchSpace(
        n_agents=n_agents,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_moead = moead.MOEAD(weights_vector=weights)
    new_moead.compile(search_space)

    assert isinstance(new_moead.T, np.ndarray)
    assert isinstance(new_moead.z, np.ndarray)
    assert new_moead.z.shape[0] == 2


def test_moead_genetic_operators():
    weights, n_agents = das_dennis(2, 2)
    search_space = SearchSpace(
        n_agents=n_agents,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_moead = moead.MOEAD(weights_vector=weights)
    new_moead.compile(search_space)

    parent1 = search_space.agents[0]
    parent2 = search_space.agents[1]

    children = new_moead._genetic_operators(parent1, parent2)

    assert len(children) > 0
    child1 = children[0]
    assert child1.position.shape == parent1.position.shape
    if len(children) == 2:
        child2 = children[1]
        assert child2.position.shape == parent2.position.shape


def test_moead_select_neighbors():
    weights, n_agents = das_dennis(2, 2)
    search_space = SearchSpace(
        n_agents=n_agents,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_moead = moead.MOEAD(weights_vector=weights)
    new_moead.compile(search_space)

    neighbors = new_moead._select_neighbors(0, search_space)

    assert isinstance(neighbors, np.ndarray)
    assert len(neighbors) == 2
    assert all(n in new_moead.T[0] for n in neighbors)


def test_moead_evaluate():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    weights, n_agents = das_dennis(2, 2)
    search_space = SearchSpace(
        n_agents=n_agents,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_moead = moead.MOEAD(weights_vector=weights)
    new_moead.compile(search_space)

    new_moead.evaluate(search_space, multi_square)

    assert isinstance(new_moead.z, np.ndarray)
    assert new_moead.z.shape[0] == 2


def test_moead_update():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    weights, n_agents = das_dennis(2, 2)
    search_space = SearchSpace(
        n_agents=n_agents,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_moead = moead.MOEAD(weights_vector=weights)
    new_moead.compile(search_space)

    new_moead.update(search_space, multi_square)
