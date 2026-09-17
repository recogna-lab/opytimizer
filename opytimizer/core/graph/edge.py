"""Edge representation for general graphs."""

from dataclasses import dataclass
from typing import Optional

from opytimizer.core.graph.node import GraphNode


@dataclass
class Edge:
    """A connection between two `GraphNode`s.

    Args:
        source: Origin node.
        target: Destination node.
        weight: Optional numeric weight/cost associated with the edge.
        directed: Whether this edge should be treated as one-way
            (`source -> target` only) or two-way.

    """

    source: GraphNode
    target: GraphNode
    weight: Optional[float] = None
    directed: bool = False

    def __post_init__(self) -> None:
        self.source.add_child(self.target)
        if not self.directed:
            self.target.add_child(self.source)
