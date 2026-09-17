"""Core abstractions for graph- and tree-based representations.
"""

from opytimizer.core.graph.agent_graph import GraphAgent
from opytimizer.core.graph.edge import Edge
from opytimizer.core.graph.generator import generate_node, generate_typed_tree
from opytimizer.core.graph.graph import Graph
from opytimizer.core.graph.node import GraphNode
from opytimizer.core.graph.operators import (
    crossover_trees,
    mutate_graph_structure,
    mutate_tree,
)
from opytimizer.core.graph.primitive import Ephemeral, Primitive, Terminal
from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.core.graph.space import (
    _MultiObjectiveSpace,
    _MultiObjectiveTensorSpace,
    _SingleObjectiveSpace,
    _SingleObjectiveTensorSpace,
)
from opytimizer.core.graph.tree import Tree
