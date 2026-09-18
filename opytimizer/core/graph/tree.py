"""Tree — a constrained `Graph` where every node has at most one parent
and there are no cycles.
"""

from typing import List, Optional, Type

from opytimizer.core.graph.graph import Graph
from opytimizer.core.graph.node import GraphNode
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Tree(Graph):
    """A rooted tree: a directed, acyclic `Graph` with a single root and at
    most one parent per node.

    Args:
        root: The tree's root node. May be `None` for an empty tree that
            will be populated later (e.g., by a generator).

    """

    def __init__(self, root: Optional[GraphNode] = None) -> None:
        super().__init__(directed=True)
        self.root = root
        if root is not None:
            self._collect(root)

    def _collect(self, node: GraphNode) -> None:
        """Registers `node` and its whole subtree into `self.nodes`."""

        self.add_node(node)
        for child in node.children:
            self._collect(child)

    @property
    def depth(self) -> int:
        """Depth of the tree (number of edges on the longest root-to-leaf path, plus one)."""

        def _depth(node: GraphNode) -> int:
            if not node.children:
                return 1
            return 1 + max(_depth(child) for child in node.children)

        return _depth(self.root) if self.root is not None else 0

    def __str__(self) -> str:
        if not self.root:
            return "Empty Tree"
        return str(self.root)

    def nodes_of_type(self, output_type: Type) -> List[GraphNode]:
        """Returns every node whose `output_type` matches — used by
        type-safe crossover/mutation to pick compatible cut points."""

        return [node for node in self.nodes if node.output_type is output_type]

    def replace_subtree(self, old_node: GraphNode, new_node: GraphNode) -> None:
        """Replaces `old_node` (and its whole subtree) with `new_node`
        in place, updating parent links and the cached node list."""

        if old_node is self.root or not old_node.parents:
            self.root = new_node
            new_node._parents = []
        else:
            parent = old_node.parents[0]
            idx = parent.children.index(old_node)
            parent.children[idx] = new_node
            new_node._parents = [parent]

        self.nodes = []
        self._collect(self.root)

    def copy(self) -> "Tree":
        return Tree(self.root.copy()) if self.root is not None else Tree()

    def evaluate(self):
        """Evaluates the whole tree by recursively evaluating its root."""

        if self.root is None:
            raise ValueError("Cannot evaluate an empty tree.")
        return self.root.evaluate()
