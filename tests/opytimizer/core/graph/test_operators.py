
from opytimizer.core.graph.node import GraphNode
from opytimizer.core.graph.operators import (
    crossover_trees,
    mutate_graph_structure,
    mutate_tree,
)
from opytimizer.core.graph.tree import Tree


def make_tree(value):
    root = GraphNode("root", output_type=int, is_terminal=False, value=lambda x: x)
    child = GraphNode(str(value), value=value, output_type=int)
    root.add_child(child, expected_type=int)
    return Tree(root)


def test_crossover_trees_is_noop_for_same_tree():
    tree = make_tree(1)
    before = tree.copy()

    crossover_trees(tree, tree)

    assert str(tree) == str(before)


def test_crossover_trees_swaps_compatible_subtrees(monkeypatch):
    tree_a = make_tree(1)
    tree_b = make_tree(2)

    monkeypatch.setattr(
        "opytimizer.core.graph.operators.random.choice",
        lambda seq: seq[0],
    )

    crossover_trees(tree_a, tree_b)

    assert tree_a.evaluate() == 2
    assert tree_b.evaluate() == 1


def test_crossover_trees_noop_without_common_types():
    root_a = GraphNode("a", value=1, output_type=int)
    root_b = GraphNode("b", value=2, output_type=float)

    tree_a = Tree(root_a)
    tree_b = Tree(root_b)

    crossover_trees(tree_a, tree_b)

    assert tree_a.evaluate() == 1
    assert tree_b.evaluate() == 2


def test_mutate_tree_replaces_selected_subtree(monkeypatch):
    from opytimizer.core.graph.primitive_set import PrimitiveSet

    pset = PrimitiveSet("int_test", int)
    pset.add_terminal(10, int, "ten")
    pset.add_terminal(20, int, "twenty")
    tree = make_tree(1)

    monkeypatch.setattr(
        "opytimizer.core.graph.operators.random.choice",
        lambda seq: seq[0],
    )

    mutate_tree(tree, pset, max_depth=1)

    assert isinstance(tree, Tree)
    assert tree.root is not None
    assert tree.root.output_type is int


def test_mutate_graph_structure_can_add_edge(monkeypatch):
    graph = make_tree(2)

    initial_edges = len(graph.edges)

    monkeypatch.setattr(
        "opytimizer.core.graph.operators.random.random",
        lambda: 0.0,
    )

    mutate_graph_structure(
        graph,
        add_edge_prob=1.0,
        remove_edge_prob=0.0,
    )

    assert len(graph.edges) > initial_edges


def test_mutate_graph_structure_can_remove_edge(monkeypatch):
    graph = make_tree(1)

    monkeypatch.setattr(
        "opytimizer.core.graph.operators.random.random",
        lambda: 0.0,
    )

    # With one existing edge, removal happens first and addition happens second.
    mutate_graph_structure(graph, add_edge_prob=0.0, remove_edge_prob=1.0)

    assert len(graph.edges) == 0


def test_mutate_graph_structure_ignores_single_node_graph():
    tree = Tree(GraphNode("only", value=1, output_type=int))

    mutate_graph_structure(tree, add_edge_prob=1.0, remove_edge_prob=1.0)

    assert len(tree.edges) == 0
