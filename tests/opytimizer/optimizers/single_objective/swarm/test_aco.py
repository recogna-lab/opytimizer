import time

import numpy as np
import pytest

from opytimizer.core.graph.graph import Graph
from opytimizer.core.graph.node import GraphNode
from opytimizer.optimizers.single_objective.swarm.aco import ACO, TSPACO
from opytimizer.utils import exception as e


def make_graph(names=("n0", "n1", "n2"), directed=False):
    graph = Graph(directed=directed)
    nodes = [GraphNode(name) for name in names]
    for node in nodes:
        graph.add_node(node)

    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i], nodes[i + 1])

    return graph


def test_aco_defaults():
    optimizer = ACO()

    assert optimizer.alpha == 1.0
    assert optimizer.beta == 5.0
    assert optimizer.rho == 0.5
    assert optimizer.q == 100.0
    assert optimizer.tau_0 == 1.0


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("alpha", -1),
        ("beta", -1),
        ("rho", -0.1),
        ("rho", 1.1),
        ("q", -1),
        ("tau_0", -1),
    ],
)
def test_aco_rejects_invalid_parameter_values(attribute, value):
    optimizer = ACO()

    with pytest.raises(e.ValueError):
        setattr(optimizer, attribute, value)


@pytest.mark.parametrize(
    "attribute",
    ["alpha", "beta", "rho", "q", "tau_0"],
)
def test_aco_rejects_invalid_parameter_types(attribute):
    optimizer = ACO()

    with pytest.raises(e.TypeError):
        setattr(optimizer, attribute, "invalid")


def test_aco_pheromone_setter_requires_numpy_array():
    optimizer = ACO()

    with pytest.raises(e.TypeError):
        optimizer.pheromone = [[1.0]]

    with pytest.raises(e.TypeError):
        optimizer.visibility = [[1.0]]


def test_aco_compile(graph_space):
    optimizer = ACO()
    optimizer.compile(graph_space)

    assert optimizer.pheromone.shape == (graph_space.n_nodes, graph_space.n_nodes)
    assert optimizer.visibility.shape == (graph_space.n_nodes, graph_space.n_nodes)
    assert np.all(optimizer.pheromone == optimizer.tau_0)
    assert np.all(optimizer.visibility == 1.0)


def test_aco_evaluate_updates_fitness_and_best_agent(graph_space):
    optimizer = ACO()

    def objective(graph):
        return float(len(graph.edges))

    old_ts = graph_space.best_agent.ts
    optimizer.evaluate(graph_space, objective)

    assert all(agent.fit == len(agent.position.edges) for agent in graph_space.agents)
    assert graph_space.best_agent.fit <= min(agent.fit for agent in graph_space.agents)
    assert graph_space.best_agent.position is not None
    assert graph_space.best_agent.ts >= old_ts


def test_aco_deposit_pheromone_undirected(graph_space):
    optimizer = ACO({"rho": 0.5, "q": 100.0, "tau_0": 1.0})
    optimizer.compile(graph_space)

    graph_space.agents[0].position = make_graph(directed=False)
    graph_space.agents[0].fit = 9.0

    graph_space.agents[1].position = make_graph(directed=False)
    graph_space.agents[1].fit = 19.0

    optimizer._deposit_pheromone(graph_space)

    expected_01 = 0.5 + 100.0 / 10.0 + 100.0 / 20.0
    assert optimizer.pheromone[0, 1] == pytest.approx(expected_01)
    assert optimizer.pheromone[1, 0] == pytest.approx(expected_01)


def test_aco_deposit_pheromone_directed(tsp_space):
    optimizer = ACO({"rho": 0.5, "q": 100.0, "tau_0": 1.0})
    optimizer.compile(tsp_space)

    tsp_space.agents[0].position = make_graph(directed=True)
    tsp_space.agents[0].fit = 9.0

    optimizer._deposit_pheromone(tsp_space)

    assert optimizer.pheromone[0, 1] == pytest.approx(0.5 + 10.0)
    assert optimizer.pheromone[1, 0] == pytest.approx(0.5)


def test_aco_construct_graph_accepts_all_edges(monkeypatch, graph_space):
    optimizer = ACO({"alpha": 1.0, "beta": 1.0})
    optimizer.compile(graph_space)

    monkeypatch.setattr(
        "opytimizer.optimizers.single_objective.swarm.aco.r.generate_uniform_random_number",
        lambda: 0.0,
    )

    graph = optimizer._construct_graph(graph_space)

    assert len(graph.nodes) == graph_space.n_nodes
    assert len(graph.edges) == graph_space.n_nodes * (graph_space.n_nodes - 1) // 2


def test_aco_construct_graph_rejects_all_edges(monkeypatch, graph_space):
    optimizer = ACO({"alpha": 1.0, "beta": 1.0})
    optimizer.compile(graph_space)

    monkeypatch.setattr(
        "opytimizer.optimizers.single_objective.swarm.aco.r.generate_uniform_random_number",
        lambda: 1.0,
    )

    graph = optimizer._construct_graph(graph_space)

    assert len(graph.nodes) == graph_space.n_nodes
    assert len(graph.edges) == 0


def test_aco_update_replaces_every_agent_position(monkeypatch, graph_space):
    optimizer = ACO()
    optimizer.compile(graph_space)

    for agent in graph_space.agents:
        agent.fit = 1.0

    monkeypatch.setattr(
        "opytimizer.optimizers.single_objective.swarm.aco.r.generate_uniform_random_number",
        lambda: 1.0,
    )

    optimizer.update(graph_space)

    assert all(agent.position is not None for agent in graph_space.agents)
    assert all(len(agent.position.edges) == 0 for agent in graph_space.agents)


@pytest.mark.parametrize(
    "distance_matrix",
    [
        None,
        [0, 1],
        np.zeros((2, 3)),
        np.zeros((2, 2, 2)),
    ],
)
def test_tspaco_rejects_invalid_distance_matrix(distance_matrix):
    with pytest.raises((e.TypeError, e.ValueError)):
        TSPACO(distance_matrix=distance_matrix)


def test_tspaco_initialization(distance_matrix):
    optimizer = TSPACO(distance_matrix=distance_matrix)

    assert np.array_equal(optimizer.distance_matrix, distance_matrix)


def test_tspaco_compile_builds_inverse_distance_visibility(distance_matrix, tsp_space):
    optimizer = TSPACO(distance_matrix=distance_matrix)
    optimizer.compile(tsp_space)

    assert optimizer.pheromone.shape == (4, 4)
    assert optimizer.visibility[0, 1] == pytest.approx(0.5)
    assert optimizer.visibility[0, 2] == pytest.approx(1 / 9)
    assert optimizer.visibility[0, 0] == 0.0


def test_tspaco_compile_rejects_wrong_space_size(distance_matrix):
    optimizer = TSPACO(distance_matrix=distance_matrix)

    wrong_space = type(
        "Space",
        (),
        {"n_nodes": 3},
    )()

    with pytest.raises(e.ValueError):
        optimizer.compile(wrong_space)


def test_tspaco_construct_graph_builds_hamiltonian_cycle(
    monkeypatch, distance_matrix, tsp_space
):
    optimizer = TSPACO(distance_matrix=distance_matrix)
    optimizer.compile(tsp_space)

    sequence = iter([1, 2, 3])
    monkeypatch.setattr(
        "numpy.random.choice",
        lambda candidates, p: next(sequence),
    )

    graph = optimizer._construct_graph(tsp_space)

    assert graph.directed is True
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4

    edge_pairs = [(edge.source.name, edge.target.name) for edge in graph.edges]
    assert edge_pairs == [
        ("n0", "n1"),
        ("n1", "n2"),
        ("n2", "n3"),
        ("n3", "n0"),
    ]


def test_tspaco_construct_graph_fallback_when_desirabilities_are_zero(
    monkeypatch, distance_matrix, tsp_space
):
    optimizer = TSPACO(distance_matrix=distance_matrix)
    optimizer.compile(tsp_space)
    optimizer.visibility[:] = 0.0

    sequence = iter([1, 2, 3])
    monkeypatch.setattr(
        "numpy.random.choice",
        lambda candidates, p: next(sequence),
    )

    graph = optimizer._construct_graph(tsp_space)

    assert len(graph.edges) == 4
    assert graph.edges[-1].target.name == "n0"


def test_tspaco_deposit_pheromone_uses_q_over_tour_length(distance_matrix, tsp_space):
    optimizer = TSPACO(
        params={"rho": 0.5, "q": 100.0},
        distance_matrix=distance_matrix,
    )
    optimizer.compile(tsp_space)

    tour = make_graph(directed=True)
    # Complete the cycle.
    tour.add_edge(tour.nodes[-1], tour.nodes[0])
    tsp_space.agents[0].position = tour
    tsp_space.agents[0].fit = 20.0

    optimizer._deposit_pheromone(tsp_space)

    assert optimizer.pheromone[0, 1] == pytest.approx(0.5 + 5.0)
    assert optimizer.pheromone[1, 2] == pytest.approx(0.5 + 5.0)
    assert optimizer.pheromone[2, 0] == pytest.approx(0.5 + 5.0)
    assert optimizer.pheromone[0, 2] == pytest.approx(0.5)


def test_tspaco_deposit_pheromone_skips_non_positive_fitness(
    distance_matrix, tsp_space
):
    optimizer = TSPACO(distance_matrix=distance_matrix)
    optimizer.compile(tsp_space)

    tsp_space.agents[0].position = make_graph(directed=True)
    tsp_space.agents[0].fit = 0.0

    before = optimizer.pheromone.copy()
    optimizer._deposit_pheromone(tsp_space)

    assert np.array_equal(optimizer.pheromone, before * optimizer.rho)
