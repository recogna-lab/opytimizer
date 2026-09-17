import pytest

from opytimizer.core.graph.node import GraphNode
from opytimizer.core.graph.tree import Tree


def test_empty_tree():
    tree = Tree()

    assert tree.root is None
    assert tree.nodes == []
    assert tree.depth == 0
    assert str(tree) == "Empty Tree"


def test_tree_collects_root_and_descendants():
    root = GraphNode("root", output_type=int)
    child = GraphNode("child", value=1, output_type=int)
    root.add_child(child)

    tree = Tree(root)

    assert tree.root is root
    assert tree.nodes == [root, child]
    assert tree.depth == 2
    assert str(tree) == "root(child)"


def test_nodes_of_type():
    root = GraphNode("root", output_type=int)
    left = GraphNode("left", value=1, output_type=int)
    right = GraphNode("right", value=True, output_type=bool)
    root.add_child(left)
    root.add_child(right)

    tree = Tree(root)

    assert tree.nodes_of_type(int) == [root, left]
    assert tree.nodes_of_type(bool) == [right]


def test_replace_subtree_non_root():
    root = GraphNode("root", output_type=int)
    old = GraphNode("old", value=1, output_type=int)
    sibling = GraphNode("sibling", value=2, output_type=int)
    new = GraphNode("new", value=3, output_type=int)

    root.add_child(old)
    root.add_child(sibling)
    tree = Tree(root)

    tree.replace_subtree(old, new)

    assert root.children == [new, sibling]
    assert new.parents == [root]
    assert old not in tree.nodes
    assert new in tree.nodes
    assert tree.root is root


def test_replace_subtree_root():
    old_root = GraphNode("old", value=1, output_type=int)
    child = GraphNode("child", value=2, output_type=int)
    old_root.add_child(child)

    tree = Tree(old_root)
    new_root = GraphNode("new", value=3, output_type=int)

    tree.replace_subtree(old_root, new_root)

    assert tree.root is new_root
    assert new_root.parents == []
    assert tree.nodes == [new_root]


def test_copy_is_independent():
    root = GraphNode("root", value=lambda x: x + 1, output_type=int, is_terminal=False)
    child = GraphNode("child", value=2, output_type=int)
    root.add_child(child)

    tree = Tree(root)
    copied = tree.copy()

    assert copied is not tree
    assert copied.root is not root
    assert copied.evaluate() == tree.evaluate()
    assert copied.nodes[1] is not tree.nodes[1]


def test_evaluate_empty_tree_raises():
    with pytest.raises(ValueError):
        Tree().evaluate()


def test_evaluate_tree():
    root = GraphNode("add", value=lambda a, b: a + b, output_type=int, is_terminal=False)
    root.add_child(GraphNode("one", value=2, output_type=int))
    root.add_child(GraphNode("two", value=4, output_type=int))

    assert Tree(root).evaluate() == 6
