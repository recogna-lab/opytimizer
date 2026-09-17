import pytest

import opytimizer.utils.exception as e
from opytimizer.core import Environment
from opytimizer.core.graph.graph import Graph
from opytimizer.spaces.graph import (
    GraphSpace,
    _MultiObjectiveGraphSpace,
    _SingleObjectiveGraphSpace,
)


def test_single_objective_graph_space():
    space = GraphSpace(
        n_agents=5,
        n_nodes=4,
        n_objectives=1,
    )

    assert isinstance(space, _SingleObjectiveGraphSpace)
    assert space.n_agents == 5
    assert space.n_nodes == 4
    assert space.n_objectives == 1

    assert len(space.agents) == 5

    for agent in space.agents:
        assert isinstance(agent.position, Graph)


def test_multi_objective_graph_space():
    space = GraphSpace(
        n_agents=5,
        n_nodes=4,
        n_objectives=2,
    )

    assert isinstance(space, _MultiObjectiveGraphSpace)
    assert space.n_agents == 5
    assert space.n_nodes == 4
    assert space.n_objectives == 2

    assert len(space.agents) == 5

    for agent in space.agents:
        assert isinstance(agent.position, Graph)


@pytest.mark.parametrize("n_objectives", [1, 2, 3])
def test_graph_space_factory(n_objectives):
    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=n_objectives,
        tensorized=False,
    )

    if n_objectives == 1:
        assert isinstance(space, _SingleObjectiveGraphSpace)
    else:
        assert isinstance(space, _MultiObjectiveGraphSpace)

    for agent in space.agents:
        assert isinstance(agent.position, Graph)


def test_graph_space_is_non_tensorized_by_default():
    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
    )

    assert isinstance(space, _SingleObjectiveGraphSpace)
    assert not hasattr(space, "X")


def test_single_objective_graph_space_parameters():
    space = GraphSpace(
        n_agents=4,
        n_nodes=6,
        n_objectives=1,
        directed=True,
        edge_prob=0.7,
        connected=True,
    )

    assert space.n_agents == 4
    assert space.n_nodes == 6
    assert space.n_objectives == 1
    assert space.directed is True
    assert space.edge_prob == 0.7
    assert space.connected is True


def test_multi_objective_graph_space_parameters():
    space = GraphSpace(
        n_agents=4,
        n_nodes=6,
        n_objectives=3,
        directed=True,
        edge_prob=0.7,
        connected=True,
    )

    assert space.n_agents == 4
    assert space.n_nodes == 6
    assert space.n_objectives == 3
    assert space.directed is True
    assert space.edge_prob == 0.7
    assert space.connected is True


@pytest.mark.parametrize("directed", [False, True])
def test_graph_space_preserves_directed(directed):
    space = GraphSpace(
        n_agents=4,
        n_nodes=5,
        n_objectives=1,
        directed=directed,
    )

    assert space.directed is directed

    for agent in space.agents:
        assert agent.position.directed is directed


@pytest.mark.parametrize("edge_prob", [0.0, 0.25, 0.5, 1.0])
def test_graph_space_preserves_edge_probability(edge_prob):
    space = GraphSpace(
        n_agents=3,
        n_nodes=5,
        n_objectives=1,
        edge_prob=edge_prob,
    )

    assert space.edge_prob == edge_prob


@pytest.mark.parametrize("connected", [False, True])
def test_graph_space_preserves_connected(connected):
    space = GraphSpace(
        n_agents=3,
        n_nodes=5,
        n_objectives=1,
        connected=connected,
    )

    assert space.connected is connected


def test_graph_space_initializes_each_agent_with_graph():
    space = GraphSpace(
        n_agents=5,
        n_nodes=4,
        n_objectives=1,
    )

    assert len(space.agents) == 5

    for agent in space.agents:
        assert isinstance(agent.position, Graph)
        assert len(agent.position.nodes) == 4


def test_graph_space_best_agent_is_graph():
    space = GraphSpace(
        n_agents=4,
        n_nodes=5,
        n_objectives=1,
    )

    assert isinstance(space.best_agent.position, Graph)


def test_graph_space_best_agent_is_copy():
    space = GraphSpace(
        n_agents=4,
        n_nodes=5,
        n_objectives=1,
    )

    assert space.best_agent.position is not space.agents[0].position


def test_graph_space_best_agent_has_same_initial_structure():
    space = GraphSpace(
        n_agents=4,
        n_nodes=5,
        n_objectives=1,
    )

    best = space.best_agent.position
    first = space.agents[0].position

    assert len(best.nodes) == len(first.nodes)
    assert len(best.edges) == len(first.edges)


def test_graph_space_default_environment():
    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
    )

    assert isinstance(space.env, Environment)
    assert space.env.backend == "numpy"


def test_graph_space_custom_environment():
    env = Environment("numpy", "float64")

    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
        env=env,
    )

    assert space.env is env


@pytest.mark.parametrize(
    "n_objectives",
    [0, -1, -2],
)
def test_graph_space_invalid_number_of_objectives(n_objectives):
    with pytest.raises(e.ValueError):
        GraphSpace(
            n_agents=3,
            n_nodes=4,
            n_objectives=n_objectives,
        )


@pytest.mark.parametrize(
    "n_nodes",
    [0, -1, -2],
)
def test_graph_space_invalid_number_of_nodes(n_nodes):
    with pytest.raises(e.ValueError):
        GraphSpace(
            n_agents=3,
            n_nodes=n_nodes,
            n_objectives=1,
        )


def test_graph_space_mapping():
    mapping = ["x", "y", "z"]

    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
        mapping=mapping,
    )

    assert space.mapping == mapping


def test_graph_space_single_objective_best_agent_fit():
    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
    )

    assert space.best_agent.fit == space.agents[0].fit


def test_graph_space_agents_have_independent_graphs():
    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
    )

    positions = [agent.position for agent in space.agents]

    assert len({id(position) for position in positions}) == len(positions)


def test_graph_space_uses_graph_random(monkeypatch):
    calls = []

    original_random = Graph.random

    def mock_random(n_nodes, edge_prob, directed, connected):
        calls.append((n_nodes, edge_prob, directed, connected))
        return original_random(
            n_nodes,
            edge_prob,
            directed,
            connected,
        )

    monkeypatch.setattr(Graph, "random", mock_random)

    GraphSpace(
        n_agents=4,
        n_nodes=5,
        n_objectives=1,
        directed=True,
        edge_prob=0.4,
        connected=True,
    )

    assert len(calls) == 4

    for call in calls:
        assert call == (5, 0.4, True, True)


def test_graph_space_tensorized_false_explicitly():
    space = GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
        tensorized=False,
    )

    assert isinstance(space, _SingleObjectiveGraphSpace)
    assert all(isinstance(agent.position, Graph) for agent in space.agents)
