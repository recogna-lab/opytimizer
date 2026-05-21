import numpy as np

from opytimizer.core.agent import Agent
from opytimizer.optimizers.multi_objective.evolutionary import rvea
from opytimizer.spaces.search import SearchSpace
from opytimizer.utils.reference_vectors import das_dennis


REFERENCE_VECTORS, N_AGENTS = das_dennis(2, 9) # 10 agents/reference vectors

def test_rvea_compile():
    search_space = SearchSpace(
        n_agents=N_AGENTS,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    try:
        new_rvea = rvea.RVEA()
    except:
        new_rvea = rvea.RVEA(reference_vectors=REFERENCE_VECTORS)
    finally:
        new_rvea.compile(search_space)



def test_rvea_crossover():
    search_space = SearchSpace(
        n_agents=N_AGENTS,
        n_variables=2,
        n_objectives=2,
        lower_bound=[1, 1],
        upper_bound=[10, 10],
    )

    new_rvea = rvea.RVEA(reference_vectors=REFERENCE_VECTORS)
    children = new_rvea.crossover_operator(search_space.agents[0], search_space.agents[1])

    alpha = children[0]
    assert type(alpha).__name__ == "Agent"
    if len(children) == 2:
        beta = children[1]
        assert type(beta).__name__ == "Agent"


def test_rvea_mutation():
    search_space = SearchSpace(
        n_agents=N_AGENTS,
        n_variables=2,
        n_objectives=2,
        lower_bound=[1, 1],
        upper_bound=[10, 10],
    )

    new_rvea = rvea.RVEA(reference_vectors=REFERENCE_VECTORS)

    alpha = new_rvea.mutation_operator(search_space.agents[0])
    beta = new_rvea.mutation_operator(search_space.agents[1])

    assert type(alpha).__name__ == "Agent"
    assert type(beta).__name__ == "Agent"


def make_agent(fit):
    a = Agent(3, 1, 2, [0, 0, 0], [1, 1, 1])
    a.fit = np.array(fit)
    return a


def test_rvea_evaluate():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=N_AGENTS,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_rvea = rvea.RVEA(reference_vectors=REFERENCE_VECTORS)
    new_rvea.compile(search_space)

    new_rvea.evaluate(search_space, multi_square)

    assert isinstance(search_space.pareto_front, list)
    assert len(search_space.pareto_front) > 0


def test_rvea_update():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=N_AGENTS,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_rvea = rvea.RVEA(reference_vectors=REFERENCE_VECTORS)
    new_rvea.compile(search_space)
    new_rvea.evaluate(space=search_space, function=multi_square)
    new_rvea.update(search_space, multi_square)


def test_rvea_reference_vectors():
    search_space = SearchSpace(
        n_agents=9,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    try:
        new_rvea = rvea.RVEA(reference_vectors=REFERENCE_VECTORS)
        new_rvea.compile(space=search_space)
    except:
        search_space = SearchSpace(
        n_agents=N_AGENTS,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )
    finally:
        assert search_space.n_agents == len(REFERENCE_VECTORS)