import numpy as np

from opytimizer.core.agent import Agent
from opytimizer.optimizers.multi_objective.evolutionary import nsga3
from opytimizer.spaces.search import SearchSpace




def test_nsga3_compile():
    search_space = SearchSpace(
        n_agents=13,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_nsga3 = nsga3.NSGA3()
    new_nsga3.compile(search_space)

    try:
        new_nsga3.rank = 1
    except:
        new_nsga3.rank = np.array([1])

    assert new_nsga3.rank == np.array([1])



def test_nsga3_crossover():
    search_space = SearchSpace(
        n_agents=10,
        n_variables=2,
        n_objectives=2,
        lower_bound=[1, 1],
        upper_bound=[10, 10],
    )

    new_nsga3 = nsga3.NSGA3()

    children = new_nsga3._crossover(search_space.agents[0], search_space.agents[1])

    alpha = children[0]
    assert type(alpha).__name__ == "Agent"
    if len(children) == 2:
        beta = children[1]
        assert type(beta).__name__ == "Agent"


def test_nsga3_mutation():
    search_space = SearchSpace(
        n_agents=10,
        n_variables=2,
        n_objectives=2,
        lower_bound=[1, 1],
        upper_bound=[10, 10],
    )

    new_nsga3 = nsga3.NSGA3()

    alpha = new_nsga3._mutation(search_space.agents[0])
    beta = new_nsga3._mutation(search_space.agents[1])

    assert type(alpha).__name__ == "Agent"
    assert type(beta).__name__ == "Agent"


def make_agent(fit):
    a = Agent(3, 1, 2, [0, 0, 0], [1, 1, 1])
    a.fit = np.array(fit)
    return a


def test_nsga3_fast_non_dominated_sort():
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5]),
        make_agent([3, 3]),
        make_agent([0.5, 2.5]),
    ]
    new_nsga3 = nsga3.NSGA3()
    fronts = new_nsga3._fast_non_dominated_sort(agents)
    assert isinstance(fronts, list)
    assert len(fronts) > 0
    assert 0 in fronts[0] or 1 in fronts[0]



def test_nsga3_tournament_selection():
    new_nsga3 = nsga3.NSGA3()
    agents = [
        make_agent([1, 2]),
        make_agent([2, 1]),
        make_agent([1.5, 1.5]),
        make_agent([3, 3]),
        make_agent([0.5, 2.5]),
    ]

    new_nsga3._fast_non_dominated_sort(agents)

    selected = new_nsga3._tournament_selection(agents)

    assert (
        isinstance(selected, list)
        or isinstance(selected, Agent)
        or isinstance(selected, int)
    )
    assert len(selected) == 5


def test_nsga3_evaluate():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=13,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_nsga3 = nsga3.NSGA3()
    new_nsga3.compile(search_space)

    new_nsga3.evaluate(search_space, multi_square)

    assert isinstance(search_space.pareto_front, list)
    assert len(search_space.pareto_front) > 0


def test_nsga3_update():
    def multi_square(x):
        f1 = np.sum(x**2)
        f2 = np.sum(x)
        return np.array([f1, f2])

    search_space = SearchSpace(
        n_agents=13,
        n_variables=2,
        n_objectives=2,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_nsga3 = nsga3.NSGA3()
    new_nsga3.compile(search_space)

    new_nsga3.update(search_space, multi_square)


def test_nsga3_conflict_number_reference_points():
    try:
        search_space = SearchSpace(
            n_agents=2,
            n_variables=2,
            n_objectives=2,
            lower_bound=[0, 0],
            upper_bound=[10, 10],
        )

        new_nsga3 = nsga3.NSGA3()
        new_nsga3.compile(search_space)
    except:
       assert len(new_nsga3.reference_points) == 13
       search_space.n_agents = 13
       new_nsga3.compile(search_space)
    
 