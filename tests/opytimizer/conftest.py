import numpy as np
import pytest

from opytimizer.core.graph.graph import Graph
from opytimizer.core.graph.node import GraphNode
from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.spaces.graph import GraphSpace


@pytest.fixture
def simple_graph():
    graph = Graph(directed=False)
    a = GraphNode("a")
    b = GraphNode("b")
    c = GraphNode("c")
    for node in (a, b, c):
        graph.add_node(node)
    graph.add_edge(a, b, weight=2.5)
    graph.add_edge(b, c, weight=3.5)
    return graph


@pytest.fixture
def directed_graph():
    graph = Graph(directed=True)
    a = GraphNode("a")
    b = GraphNode("b")
    c = GraphNode("c")
    for node in (a, b, c):
        graph.add_node(node)
    graph.add_edge(a, b, weight=1.0)
    graph.add_edge(b, c, weight=2.0)
    return graph


@pytest.fixture
def primitive_set():
    pset = PrimitiveSet("test", float)
    pset.add_terminal(1.0, float, "one")
    pset.add_terminal(2.0, float, "two")
    pset.add_primitive(lambda a, b: a + b, (float, float), float, "add")
    return pset


@pytest.fixture
def graph_space():
    return GraphSpace(
        n_agents=3,
        n_nodes=4,
        n_objectives=1,
        directed=False,
        edge_prob=0.5,
    )


@pytest.fixture
def tsp_space():
    return GraphSpace(
        n_agents=2,
        n_nodes=4,
        n_objectives=1,
        directed=True,
        edge_prob=0.5,
    )


@pytest.fixture
def distance_matrix():
    return np.array(
        [
            [0.0, 2.0, 9.0, 10.0],
            [2.0, 0.0, 6.0, 4.0],
            [9.0, 6.0, 0.0, 8.0],
            [10.0, 4.0, 8.0, 0.0],
        ]
    )
