import numpy as np

from opytimizer.core.agent import Agent
from opytimizer.optimizers.multi_objective.evolutionary import rvea
from opytimizer.spaces.search import SearchSpace
from opytimizer.utils.reference_vectors import das_dennis

REFERENCE_VECTORS, N_AGENTS = das_dennis(2, 9)  # 10 agents/reference vectors


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
    child1, child2 = new_rvea.crossover_operator(
        search_space.agents[0], search_space.agents[1]
    )

    alpha = child1[0]
    assert type(alpha).__name__ == "Agent"


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
    search_space.update_pareto_front()
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


def _dtlz2(x):
    x = np.asarray(x).reshape(-1)
    g = np.sum((x[2:] - 0.5) ** 2)
    a, b = x[0] * np.pi / 2, x[1] * np.pi / 2
    return np.array(
        [
            (1 + g) * np.cos(a) * np.cos(b),
            (1 + g) * np.cos(a) * np.sin(b),
            (1 + g) * np.sin(a),
        ]
    )


def test_rvea_default_operators_follow_the_paper():
    space = SearchSpace(N_AGENTS, 6, 2, [0.0] * 6, [1.0] * 6)
    new_rvea = rvea.RVEA(reference_vectors=REFERENCE_VECTORS)
    new_rvea.compile(space)

    assert new_rvea.crossover_operator.eta == 30
    assert new_rvea.crossover_operator.rate == 1.0
    assert new_rvea.mutation_operator.eta == 20
    assert new_rvea.mutation_operator.rate == 1.0 / 6


def test_rvea_population_varies_as_in_the_paper():
    np.random.seed(0)
    weights, n = das_dennis(3, 6)
    space = SearchSpace(n, 8, 3, [0.0] * 8, [1.0] * 8)
    new_rvea = rvea.RVEA(reference_vectors=weights, max_generations=20)
    new_rvea.compile(space)
    new_rvea.evaluate(space, _dtlz2)

    sizes = []
    for _ in range(20):
        new_rvea.update(space, _dtlz2)
        sizes.append(len(space.agents))

        # One elitist per non-empty subpopulation, all of them distinct
        assert 1 <= len(space.agents) <= n
        assert len({id(a) for a in space.agents}) == len(space.agents)

    # Empty subpopulations do happen, so the population size really varies
    assert min(sizes) < n


def test_rvea_tensor_population_size_is_constant():
    from opytimizer.core import Environment

    np.random.seed(0)
    env = Environment("numpy", "float64")
    weights, n = das_dennis(3, 6)
    space = SearchSpace(n, 8, 3, [0.0] * 8, [1.0] * 8, env=env, tensorized=True)

    def batch(X, xp=np):
        return np.stack([_dtlz2(x) for x in X.reshape(X.shape[0], -1)])

    new_rvea = rvea.RVEATensor(reference_vectors=weights, max_generations=20)
    new_rvea.compile(space)
    new_rvea.evaluate(space, batch)

    for _ in range(20):
        new_rvea.update(space, batch)
        k = int(new_rvea._n_survivors)

        assert space.X.shape[0] == space.F.shape[0] == n
        # Rows beyond the K survivors repeat survivors
        survivors = {tuple(f) for f in space.F[:k]}
        assert all(tuple(f) in survivors for f in space.F[k:])
