"""Type-safe genetic operators for typed trees, plus structural mutation
helpers for general graphs.
"""

import random
from collections import defaultdict
from typing import Dict, List, Type

from opytimizer.core.graph.generator import generate_node
from opytimizer.core.graph.graph import Graph
from opytimizer.core.graph.node import GraphNode
from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.core.graph.tree import Tree


def _index_by_type(tree: Tree) -> Dict[Type, List[GraphNode]]:
    index: Dict[Type, List[GraphNode]] = defaultdict(list)
    for node in tree.nodes:
        index[node.output_type].append(node)
    return index


def crossover_trees(tree_a: Tree, tree_b: Tree) -> None:
    """Swaps two randomly chosen, type-compatible subtrees between
    `tree_a` and `tree_b`, in place. No-ops if the trees share no common
    node type."""
    if tree_a is tree_b:
        return

    idx_a, idx_b = _index_by_type(tree_a), _index_by_type(tree_b)
    common_types = [t for t in idx_a if idx_b.get(t)]
    if not common_types:
        return

    output_type = random.choice(common_types)
    node_a = random.choice(idx_a[output_type])
    node_b = random.choice(idx_b[output_type])

    node_a_copy, node_b_copy = node_a.copy(), node_b.copy()
    tree_a.replace_subtree(node_a, node_b_copy)
    tree_b.replace_subtree(node_b, node_a_copy)


def mutate_tree(tree: Tree, pset: PrimitiveSet, max_depth: int = 4) -> None:
    """Replaces a randomly chosen subtree with a freshly generated one of
    the same output type, in place."""

    idx = _index_by_type(tree)
    output_type = random.choice(list(idx.keys()))
    target = random.choice(idx[output_type])
    new_subtree = generate_node(pset, output_type, max_depth, grow=True)
    tree.replace_subtree(target, new_subtree)


def mutate_graph_structure(
    graph: Graph,
    add_edge_prob: float = 0.1,
    remove_edge_prob: float = 0.1,
) -> None:
    """Randomly adds/removes edges from a general (non-tree) graph — used
    as a structural perturbation operator for `GraphSpace`'s non-tensorized
    variant."""

    nodes = graph.nodes
    if len(nodes) < 2:
        return

    if graph.edges and random.random() < remove_edge_prob:
        graph.remove_edge(random.choice(graph.edges))

    if random.random() < add_edge_prob:
        source, target = random.sample(nodes, 2)
        graph.add_edge(source, target)
