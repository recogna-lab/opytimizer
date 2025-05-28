import numpy as np
from opytimizer.optimizers.multi_objective.evolutionary import moead
from opytimizer.spaces.search import SearchSpace


def test_moead_params():
    params = {
        "CR": 0.9,
        "MR": 0.05,
        "n_subproblems": 100,
        "neighborhood_size": 10
    }

    new_moead = moead.MOEAD(params=params)
    
    assert new_moead.CR == 0.9
    assert new_moead.MR == 0.05
    assert new_moead.n_subproblems == 100
    assert new_moead.neighborhood_size == 10


def test_moead_params_setter():
    new_moead = moead.MOEAD()

    try:
        new_moead.CR = "a"
    except:
        new_moead.CR = 0.9

    try:
        new_moead.CR = -1
    except:
        new_moead.CR = 0.9

    assert new_moead.CR == 0.9

    try:
        new_moead.MR = "b"
    except:
        new_moead.MR = 0.05

    try:
        new_moead.MR = -1
    except:
        new_moead.MR = 0.05

    assert new_moead.MR == 0.05

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


def test_moead_compile():
    search_space = SearchSpace(
        n_agents=2, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead = moead.MOEAD()
    new_moead.compile(search_space)
    
    assert isinstance(new_moead.T, np.ndarray)
    assert isinstance(new_moead.z, np.ndarray)
    assert new_moead.z.shape[0] == 2


def test_moead_genetic_operators():
    search_space = SearchSpace(
        n_agents=10, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead = moead.MOEAD()
    new_moead.compile(search_space)
    
    parent1 = search_space.agents[0].position
    parent2 = search_space.agents[1].position
    
    child1, child2 = new_moead._genetic_operators(parent1, parent2, search_space)
    
    assert isinstance(child1, np.ndarray)
    assert isinstance(child2, np.ndarray)
    assert child1.shape == parent1.shape
    assert child2.shape == parent2.shape


def test_moead_select_neighbors():
    search_space = SearchSpace(
        n_agents=10, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead = moead.MOEAD()
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

    search_space = SearchSpace(
        n_agents=2, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead = moead.MOEAD()
    new_moead.compile(search_space)
    
    new_moead.evaluate(search_space, multi_square)
    
    assert isinstance(new_moead.z, np.ndarray)
    assert new_moead.z.shape[0] == 2


def test_moead_update():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])
    
    search_space = SearchSpace(
        n_agents=2, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead = moead.MOEAD()
    new_moead.compile(search_space)
    
    new_moead.update(search_space, multi_square)
