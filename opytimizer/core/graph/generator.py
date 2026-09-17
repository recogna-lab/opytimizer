"""Generation routines for typed trees."""

from typing import Type

import numpy.random as random

import opytimizer.utils.exception as e
from opytimizer.core.graph.node import GraphNode
from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.core.graph.tree import Tree


def generate_typed_tree(
    pset: PrimitiveSet,
    min_depth: int = 2,
    max_depth: int = 6,
    method: str = "half_and_half",
) -> Tree:
    """Builds a single typed tree using DEAP-style grow/full/ramped
    half-and-half generation, respecting `pset`'s type constraints.

    Args:
        pset: Registry of typed primitives/terminals to draw from.
        min_depth: Minimum tree depth.
        max_depth: Maximum tree depth.
        method: One of `grow`, `full` or `half_and_half`.

    """

    if method not in ("grow", "full", "half_and_half"):
        raise e.ValueError("`method` should be `grow`, `full` or `half_and_half`.")
    if min_depth > max_depth:
        raise e.ValueError("`min_depth` should not be greater than `max_depth`.")

    depth = random.randint(min_depth, max_depth)
    grow = method == "grow" or (method == "half_and_half" and random.random() < 0.5)

    root = generate_node(pset, pset.root_type, depth, grow)
    return Tree(root)


def generate_node(
    pset: PrimitiveSet, required_type: Type, depth_left: int, grow: bool
) -> GraphNode:
    """Recursively generates a single (sub)tree of type `required_type`.

    Args:
        pset: Registry of typed primitives/terminals to draw from.
        required_type: Type that the returned node must produce.
        depth_left: Remaining depth budget for this branch.
        grow: If `True` (grow method), leaves may be chosen early even
            when a primitive would still fit; if `False` (full method),
            leaves are only chosen once `depth_left` runs out or no
            primitive of `required_type` exists.

    """

    terminals = pset.terminals_of(required_type)
    primitives = pset.primitives_of(required_type)

    use_terminal = (
        depth_left <= 0
        or not primitives
        or (grow and terminals and random.random() < 0.3)
    )

    if use_terminal:
        if not terminals:
            raise e.ValueError(
                f"No terminal registered in `{pset.name}` for type `{required_type}` "
                "(and no primitive fits either — call `pset.validate()` to catch "
                "this kind of grammar dead-end earlier)."
            )
        term = random.choice(terminals)
        value = term.generator() if hasattr(term, "generator") else term.value
        return GraphNode(
            term.name, value=value, output_type=required_type, is_terminal=True
        )

    primitive = random.choice(primitives)
    node = GraphNode(
        primitive.name,
        value=primitive.function,
        output_type=primitive.output_type,
        is_terminal=False,
    )
    for input_type in primitive.input_types:
        child = generate_node(pset, input_type, depth_left - 1, grow)
        node.add_child(child, expected_type=input_type)
    return node
