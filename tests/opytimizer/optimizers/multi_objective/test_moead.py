import numpy as np
from opytimizer.optimizers.multi_objective.evolutionary import moead
from opytimizer.spaces.search import SearchSpace
from opytimizer.core.agent import Agent
from opytimizer.utils.decomposition import weighted_sum

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

def test_moead_compile():
    search_space = SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead = moead.MOEAD()
    new_moead.compile(search_space)
    
    assert isinstance(new_moead.T, np.ndarray)
    assert isinstance(new_moead.z, np.ndarray)

def test_moead_genetic_operators():
    search_space = SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_moead = moead.MOEAD()
    new_moead.compile(search_space)

    child1, child2 = new_moead._genetic_operators(
        search_space.agents[0].position,
        search_space.agents[1].position,
        search_space
    )

    assert isinstance(child1, np.ndarray)
    assert isinstance(child2, np.ndarray)

def test_moead_select_neighbors():
    search_space = SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_moead = moead.MOEAD()
    new_moead.compile(search_space)

    neighbors = new_moead._select_neighbors(0)

    assert isinstance(neighbors, np.ndarray)
    assert len(neighbors) == 2

def test_moead_dominance_between_two_points():
    new_moead = moead.MOEAD()
    
    point1 = np.array([1, 2])
    point2 = np.array([2, 3])
    
    assert new_moead._dominance_between_two_points(point1, point2) == True
    assert new_moead._dominance_between_two_points(point2, point1) == False

def test_moead_evaluate():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead = moead.MOEAD()
    new_moead.compile(search_space)
    
    new_moead.evaluate(search_space, multi_square)
    
    assert isinstance(new_moead.z, np.ndarray)
    assert len(new_moead.z) == 2 