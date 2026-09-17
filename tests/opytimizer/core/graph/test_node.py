import pytest

from opytimizer.core.graph.node import GraphNode


def test_node_initialization():
    node = GraphNode("x", value=10, output_type=int, is_terminal=True)

    assert node.name == "x"
    assert node.value == 10
    assert node.output_type is int
    assert node.is_terminal is True
    assert node.children == []
    assert node.parents == []
    assert node.arity == 0
    assert repr(node) == "GraphNode(x)"
    assert str(node) == "x"


def test_add_child_updates_both_sides():
    parent = GraphNode("parent")
    child = GraphNode("child")

    parent.add_child(child)

    assert parent.children == [child]
    assert child.parents == [parent]
    assert parent.arity == 1


def test_add_child_validates_expected_type():
    parent = GraphNode("parent")
    child = GraphNode("child", output_type=int)

    with pytest.raises(TypeError):
        parent.add_child(child, expected_type=float)


def test_remove_child_updates_both_sides():
    parent = GraphNode("parent")
    child = GraphNode("child")

    parent.add_child(child)
    parent.remove_child(child)

    assert parent.children == []
    assert child.parents == []
    assert parent.arity == 0


def test_remove_missing_child_raises():
    parent = GraphNode("parent")
    child = GraphNode("child")

    with pytest.raises(ValueError):
        parent.remove_child(child)


def test_string_representation_is_recursive():
    root = GraphNode("add")
    left = GraphNode("1")
    right = GraphNode("2")
    root.add_child(left)
    root.add_child(right)

    assert str(root) == "add(1, 2)"


def test_evaluate_terminal():
    node = GraphNode("answer", value=42, is_terminal=True)

    assert node.evaluate() == 42


def test_evaluate_function_node():
    root = GraphNode("add", value=lambda a, b: a + b, is_terminal=False)
    root.add_child(GraphNode("one", value=2))
    root.add_child(GraphNode("two", value=3))

    assert root.evaluate() == 5


def test_copy_is_deep_and_preserves_structure():
    root = GraphNode("add", value=lambda a, b: a + b, is_terminal=False)
    left = GraphNode("one", value=1)
    right = GraphNode("two", value=2)
    root.add_child(left)
    root.add_child(right)

    copied = root.copy()

    assert copied is not root
    assert copied.name == root.name
    assert copied.value is root.value
    assert copied.children[0] is not left
    assert copied.children[1] is not right
    assert copied.evaluate() == 3
