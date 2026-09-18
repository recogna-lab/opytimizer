import random

import pytest

from opytimizer.core.graph.generator import generate_node, generate_typed_tree
from opytimizer.core.graph.tree import Tree
from opytimizer.utils import exception as e


def test_generate_typed_tree_returns_tree(primitive_set):
    random.seed(0)

    tree = generate_typed_tree(primitive_set, min_depth=1, max_depth=2)

    assert isinstance(tree, Tree)
    assert tree.root is not None
    assert tree.root.output_type is float
    assert tree.depth >= 1
    assert tree.depth <= 3


def test_generate_typed_tree_methods(primitive_set):
    for method in ("grow", "full", "half_and_half"):
        random.seed(1)
        tree = generate_typed_tree(primitive_set, 1, 2, method)

        assert isinstance(tree, Tree)
        assert tree.root.output_type is float


def test_generate_typed_tree_rejects_invalid_method(primitive_set):
    with pytest.raises(e.ValueError):
        generate_typed_tree(primitive_set, method="invalid")


def test_generate_typed_tree_rejects_invalid_depth_range(primitive_set):
    with pytest.raises(e.ValueError):
        generate_typed_tree(primitive_set, min_depth=3, max_depth=2)


def test_generate_node_rejects_dead_end_grammar():
    from opytimizer.core.graph.primitive_set import PrimitiveSet

    pset = PrimitiveSet("broken", float)
    pset.add_primitive(lambda x: x, (int,), float, "needs_int")

    with pytest.raises(e.ValueError):
        generate_node(pset, float, depth_left=1, grow=False)
