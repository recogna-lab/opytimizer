import random

from opytimizer.core.graph.graph import Graph
from opytimizer.core.graph.node import GraphNode


def test_graph_initialization():
    graph = Graph()

    assert graph.directed is False
    assert graph.nodes == []
    assert graph.edges == []


def test_add_node(simple_graph):
    node = GraphNode("d")
    simple_graph.add_node(node)

    assert node in simple_graph.nodes


def test_add_edge_undirected_updates_neighbors(simple_graph):
    a, b = simple_graph.nodes[0], simple_graph.nodes[1]

    assert b in simple_graph.neighbors(a)
    assert a in simple_graph.neighbors(b)
    assert len(simple_graph.edges) == 2


def test_add_edge_directed_updates_only_forward_neighbor(directed_graph):
    a, b = directed_graph.nodes[0], directed_graph.nodes[1]

    assert b in directed_graph.neighbors(a)
    assert a not in directed_graph.neighbors(b)


def test_remove_edge_undirected_updates_neighbors(simple_graph):
    edge = simple_graph.edges[0]
    source, target = edge.source, edge.target

    simple_graph.remove_edge(edge)

    assert edge not in simple_graph.edges
    assert target not in source.children
    assert source not in target.children


def test_remove_edge_directed_updates_neighbors(directed_graph):
    edge = directed_graph.edges[0]
    source, target = edge.source, edge.target

    directed_graph.remove_edge(edge)

    assert target not in source.children
    assert source not in target.children


def test_copy_is_deep_and_preserves_edges(simple_graph):
    copied = simple_graph.copy()

    assert copied is not simple_graph
    assert copied.directed == simple_graph.directed
    assert len(copied.nodes) == len(simple_graph.nodes)
    assert len(copied.edges) == len(simple_graph.edges)

    for original, clone in zip(simple_graph.nodes, copied.nodes):
        assert clone is not original
        assert clone.name == original.name

    for original_edge, cloned_edge in zip(simple_graph.edges, copied.edges):
        assert cloned_edge is not original_edge
        assert cloned_edge.weight == original_edge.weight


def test_random_graph_has_requested_number_of_nodes():
    random.seed(0)
    graph = Graph.random(8, edge_prob=0.0)

    assert len(graph.nodes) == 8
    assert len(graph.edges) == 0
    assert [node.name for node in graph.nodes] == [f"n{i}" for i in range(8)]


def test_random_undirected_graph_contains_no_self_loops():
    random.seed(0)
    graph = Graph.random(8, edge_prob=1.0, directed=False)

    assert len(graph.edges) == 28
    assert all(edge.source is not edge.target for edge in graph.edges)


def test_random_directed_graph_contains_no_self_loops():
    random.seed(0)
    graph = Graph.random(5, edge_prob=1.0, directed=True)

    assert len(graph.edges) == 20
    assert all(edge.source is not edge.target for edge in graph.edges)


def test_random_connected_undirected_graph_is_connected():
    random.seed(0)
    graph = Graph.random(10, edge_prob=0.0, connected=True)

    visited = {graph.nodes[0]}
    changed = True
    while changed:
        changed = False
        for node in list(visited):
            for neighbor in node.children:
                if neighbor not in visited:
                    visited.add(neighbor)
                    changed = True

    assert len(visited) == len(graph.nodes)
    assert len(graph.edges) >= len(graph.nodes) - 1


def test_random_connected_directed_graph_reaches_every_node_from_root():
    random.seed(0)
    graph = Graph.random(10, edge_prob=0.0, directed=True, connected=True)

    visited = {graph.nodes[0]}
    changed = True
    while changed:
        changed = False
        for node in list(visited):
            for neighbor in node.children:
                if neighbor not in visited:
                    visited.add(neighbor)
                    changed = True

    assert len(visited) == len(graph.nodes)
