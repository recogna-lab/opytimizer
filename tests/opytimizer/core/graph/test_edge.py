from opytimizer.core.graph.edge import Edge
from opytimizer.core.graph.node import GraphNode


def test_undirected_edge_links_both_nodes():
    source = GraphNode("source")
    target = GraphNode("target")

    edge = Edge(source, target, weight=2.5, directed=False)

    assert edge.source is source
    assert edge.target is target
    assert edge.weight == 2.5
    assert edge.directed is False
    assert target in source.children
    assert source in target.children


def test_directed_edge_links_only_source_to_target():
    source = GraphNode("source")
    target = GraphNode("target")

    Edge(source, target, directed=True)

    assert target in source.children
    assert source not in target.children
