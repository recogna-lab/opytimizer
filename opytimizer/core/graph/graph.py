"""General-purpose graph container (nodes + edges), directed or undirected."""

import random
from typing import Callable, Dict, List, Optional

from opytimizer.core.graph.edge import Edge
from opytimizer.core.graph.node import GraphNode
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Graph:
    """Holds a collection of `GraphNode`s and `Edge`s.

    Args:
        directed: Whether edges should be interpreted as one-way.

    """

    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self.nodes: List[GraphNode] = []
        self.edges: List[Edge] = []

    def add_node(self, node: GraphNode) -> None:
        self.nodes.append(node)

    def add_edge(
        self, source: GraphNode, target: GraphNode, weight: Optional[float] = None
    ) -> None:
        edge = Edge(source, target, weight, directed=self.directed)
        self.edges.append(edge)

    def remove_edge(self, edge: Edge) -> None:
        self.edges.remove(edge)
        edge.source.remove_child(edge.target)
        if not edge.directed:
            edge.target.remove_child(edge.source)

    def neighbors(self, node: GraphNode) -> List[GraphNode]:
        return node.children

    def copy(self) -> "Graph":
        """Deep-copies the graph, preserving node identity via a mapping
        table (so shared references — e.g., a DAG's converging edges —
        survive the copy)."""

        mapping: Dict[GraphNode, GraphNode] = {}

        def clone(n: GraphNode) -> GraphNode:
            if n not in mapping:
                mapping[n] = GraphNode(n.name, n.value, n.output_type, n.is_terminal)
            return mapping[n]

        new_graph = Graph(self.directed)
        for n in self.nodes:
            new_graph.add_node(clone(n))
        for edge in self.edges:
            new_graph.add_edge(clone(edge.source), clone(edge.target), edge.weight)
        return new_graph

    @classmethod
    def random(
        cls,
        n_nodes: int,
        edge_prob: float = 0.3,
        directed: bool = False,
        connected: bool = False,
        node_factory: Callable[[int], GraphNode] = lambda i: GraphNode(name=f"n{i}"),
    ) -> "Graph":
        """Generates an Erdos-Renyi-style random graph.

        Args:
            n_nodes: Number of nodes to create.
            edge_prob: Probability of an edge existing between any two
                nodes.
            directed: Whether generated edges are one-way.
            connected: Whether the generated graph must be connected.
            node_factory: Callable that builds the i-th node — override to
                attach custom payloads/types per node.

        """

        graph = cls(directed)
        nodes = [node_factory(i) for i in range(n_nodes)]
        for node in nodes:
            graph.add_node(node)

        if not connected:

            for i in range(n_nodes):

                j_range = range(n_nodes) if directed else range(i + 1, n_nodes)

                for j in j_range:

                    if i != j and random.random() < edge_prob:
                        graph.add_edge(nodes[i], nodes[j])

            return graph

        # Connected graphs

        for i in range(1, n_nodes):

            parent = random.randrange(i)

            if directed:
                graph.add_edge(nodes[parent], nodes[i])

            else:
                graph.add_edge(nodes[parent], nodes[i])

        # random edges

        for i in range(n_nodes):

            j_range = range(n_nodes) if directed else range(i + 1, n_nodes)

            for j in j_range:

                if i == j:
                    continue

                # Verify if the edge already exist
                exists = any(
                    edge.source is nodes[i] and edge.target is nodes[j]
                    for edge in graph.edges
                )

                if exists:
                    continue

                if random.random() < edge_prob:

                    graph.add_edge(nodes[i], nodes[j])

        return graph
