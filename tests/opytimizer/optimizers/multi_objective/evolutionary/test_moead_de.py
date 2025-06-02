import numpy as np
from opytimizer.optimizers.multi_objective.evolutionary import moead
from opytimizer.spaces.search import SearchSpace
from opytimizer.utils.weights_vector import ref_dirs

def test_moead_de_params():
    params = {
        "CR": 0.9,
        "MR": 0.05,
        "n_subproblems": 100,
        "neighborhood_size": 10,
        "nr": 2,
        "F": 0.5
    }

    new_moead_de = moead.MOEAD_DE(params=params)
    
    assert new_moead_de.CR == 0.9
    assert new_moead_de.MR == 0.05
    assert new_moead_de.n_subproblems == 100
    assert new_moead_de.neighborhood_size == 10
    assert new_moead_de.nr == 2
    assert new_moead_de.F == 0.5


def test_moead_de_params_setter():
    new_moead_de = moead.MOEAD_DE()

    try:
        new_moead_de.CR = "a"
    except:
        new_moead_de.CR = 0.9

    try:
        new_moead_de.CR = -1
    except:
        new_moead_de.CR = 0.9

    assert new_moead_de.CR == 0.9

    try:
        new_moead_de.MR = "b"
    except:
        new_moead_de.MR = 0.05

    try:
        new_moead_de.MR = -1
    except:
        new_moead_de.MR = 0.05

    assert new_moead_de.MR == 0.05

    try:
        new_moead_de.n_subproblems = "c"
    except:
        new_moead_de.n_subproblems = 100

    try:
        new_moead_de.n_subproblems = 0
    except:
        new_moead_de.n_subproblems = 100

    assert new_moead_de.n_subproblems == 100

    try:
        new_moead_de.neighborhood_size = "d"
    except:
        new_moead_de.neighborhood_size = 10

    try:
        new_moead_de.neighborhood_size = 1
    except:
        new_moead_de.neighborhood_size = 10

    assert new_moead_de.neighborhood_size == 10
    
    try:
        new_moead_de.nr = "e" 
    except:
        new_moead_de.nr = 3
        
    assert new_moead_de.nr == 3
    
    try:
        new_moead_de.nr = 0
    except:
        new_moead_de.nr = 3
    
    assert new_moead_de.nr == 3
    
    try:
        new_moead_de.F = 'f'
    except:
        new_moead_de.F = 0.5
        
    assert new_moead_de.F == 0.5
    
    try:
        new_moead_de.F = -0.5
    except:
        new_moead_de.F = 0.5
        
    assert new_moead_de.F == 0.5
    


def test_moead_de_compile():
    weights, n_agents = ref_dirs(2, 99)
    search_space = SearchSpace(
        n_agents=n_agents, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead_de = moead.MOEAD_DE(weights_vector=weights)
    new_moead_de.compile(search_space)
    
    assert isinstance(new_moead_de.z, np.ndarray)
    assert new_moead_de.z.shape[0] == 2


def test_moead_de_operators():
    weights, n_agents = ref_dirs(2, 99)
    search_space = SearchSpace(
        n_agents=n_agents, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead_de = moead.MOEAD_DE(weights_vector=weights)
    new_moead_de.compile(search_space)
    
    parent1 = search_space.agents[0].position
    parent2 = search_space.agents[1].position
    parent3 = search_space.agents[2].position
    
    child1 = new_moead_de._apply_operators(parent1, parent2, parent3, search_space)
    
    assert isinstance(child1, np.ndarray)
    assert child1.shape == parent1.shape
    


def test_moead_de_select_neighbors():
    weights, n_agents = ref_dirs(2, 99)
    search_space = SearchSpace(
        n_agents=n_agents, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead_de = moead.MOEAD_DE(weights_vector=weights)
    new_moead_de.compile(search_space)
    
    neighbors = new_moead_de._select_neighbors([0,1,2,3,4,5], search_space)
    
    assert isinstance(neighbors, np.ndarray)
    assert len(neighbors) == 3
    assert all(n in new_moead_de.T[0] for n in neighbors)


def test_moead_de_evaluate():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    weights, n_agents = ref_dirs(2, 99)
    search_space = SearchSpace(
        n_agents=n_agents, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead_de = moead.MOEAD_DE(weights_vector=weights)
    new_moead_de.compile(search_space)
    
    new_moead_de.evaluate(search_space, multi_square)
    
    assert isinstance(new_moead_de.z, np.ndarray)
    assert new_moead_de.z.shape[0] == 2


def test_moead_de_update():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])
    
    weights, n_agents = ref_dirs(2, 99)
    search_space = SearchSpace(
        n_agents=n_agents, n_variables=2, n_objectives=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )
    
    new_moead_de = moead.MOEAD_DE(weights_vector=weights)
    new_moead_de.compile(search_space)
    
    new_moead_de.update(search_space, multi_square)
