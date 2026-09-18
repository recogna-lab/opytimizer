"""Graph node representation.

A `GraphNode` is the atomic unit shared by both general graphs and typed
trees. It optionally carries an `output_type`, which is what allows a
node to participate in strongly-typed genetic programming (see
`opytimizer.core.graph.primitive_set.PrimitiveSet`). Untyped usage
(`output_type=None`) remains valid for plain, non-GP graphs.
"""

from typing import Any, List, Optional, Type

from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GraphNode:
    """A single vertex of a `Graph`/`Tree`.

    Args:
        name: Display/identifier name (e.g., the primitive's name, or a
            variable's name).
        value: Payload — for a terminal, the actual value; for a function
            node, the callable that should be applied to the node's
            children's evaluated outputs.
        output_type: Type produced when this node is evaluated. `None`
            means "untyped" (kept for plain graphs that don't need
            strongly-typed GP semantics).
        is_terminal: Whether this node is a leaf (terminal) or an internal
            function node.

    """

    def __init__(
        self,
        name: str,
        value: Any = None,
        output_type: Optional[Type] = None,
        is_terminal: bool = True,
    ) -> None:
        self.name = name
        self.value = value
        self.output_type = output_type
        self.is_terminal = is_terminal

        self._children: List["GraphNode"] = []
        self._parents: List["GraphNode"] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    def children(self) -> List["GraphNode"]:
        """Outgoing neighbors (child nodes)."""

        return self._children

    @property
    def parents(self) -> List["GraphNode"]:
        """Incoming neighbors (parent nodes). A tree node has at most one."""

        return self._parents

    @property
    def arity(self) -> int:
        """Number of children currently attached."""

        return len(self._children)

    def __str__(self) -> str:
        # Leaf Node
        if not self.children:
            return str(self.name)

        children_str = ", ".join(str(child) for child in self.children)
        return f"{self.name}({children_str})"

    def add_child(
        self, child: "GraphNode", expected_type: Optional[Type] = None
    ) -> None:
        """Attaches `child` to this node.

        Args:
            child: Node to attach.
            expected_type: If provided (typed mode), validates that
                `child.output_type` matches before attaching.

        """

        if expected_type is not None and child.output_type is not expected_type:

            raise TypeError(
                f"Expected child of type `{expected_type}`, got `{child.output_type}` "
                f"for node `{self.name}`."
            )
        self._children.append(child)
        child._parents.append(self)

    def remove_child(self, child: "GraphNode") -> None:
        """Detaches `child` from this node (and vice-versa)."""

        self._children.remove(child)
        child._parents.remove(self)

    def copy(self) -> "GraphNode":
        """Performs a deep, structure-preserving copy of the subtree/subgraph
        rooted at this node (recursively copies children)."""

        new_node = GraphNode(self.name, self.value, self.output_type, self.is_terminal)
        for child in self._children:
            new_node.add_child(child.copy(), expected_type=child.output_type)
        return new_node

    def evaluate(self) -> Any:
        """Recursively evaluates this node: terminals return their value;
        function nodes apply `self.value` (a callable) over the evaluated
        children."""

        if self.is_terminal:
            return self.value
        args = [child.evaluate() for child in self._children]
        return self.value(*args)
