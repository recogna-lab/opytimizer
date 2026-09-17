import numpy as np
import pytest

from opytimizer.core.graph.agent_graph import GraphAgent
from opytimizer.core.graph.graph import Graph
from opytimizer.utils import exception as e


def test_graph_agent_defaults():
    agent = GraphAgent()

    assert agent.n_objectives == 1
    assert agent.mapping == ["graph"]
    assert agent.position is None
    assert isinstance(agent.ts, int)
    assert agent.fit == np.finfo(float).max


def test_graph_agent_accepts_graph_position():
    agent = GraphAgent()
    graph = Graph()

    agent.position = graph

    assert agent.position is graph


def test_graph_agent_rejects_invalid_position():
    agent = GraphAgent()

    with pytest.raises(e.TypeError):
        agent.position = "not a graph"


def test_graph_agent_rejects_invalid_n_objectives():
    with pytest.raises(e.TypeError):
        GraphAgent(n_objectives=1.0)

    with pytest.raises(e.ValueError):
        GraphAgent(n_objectives=0)


def test_graph_agent_rejects_invalid_timestamp():
    agent = GraphAgent()

    with pytest.raises(e.TypeError):
        agent.ts = 1.5


def test_graph_agent_dominates():
    a = GraphAgent(n_objectives=2)
    b = GraphAgent(n_objectives=2)

    a.fit = np.array([1.0, 2.0])
    b.fit = np.array([2.0, 3.0])

    assert a.dominates(b)
    assert not b.dominates(a)


def test_graph_agent_equal_fitness_does_not_dominate():
    a = GraphAgent(n_objectives=2)
    b = GraphAgent(n_objectives=2)

    a.fit = np.array([1.0, 2.0])
    b.fit = np.array([1.0, 2.0])

    assert not a.dominates(b)
    assert not b.dominates(a)
