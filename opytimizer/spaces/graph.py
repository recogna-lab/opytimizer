"""Graph-based search spaces.
"""

import copy
from typing import List, Optional, Union

import opytimizer.utils.exception as e
from opytimizer.core import Environment
from opytimizer.core.graph import Graph
from opytimizer.core.graph.space import (
    _MultiObjectiveSpace,
    _MultiObjectiveTensorSpace,
    _SingleObjectiveSpace,
    _SingleObjectiveTensorSpace,
)
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class _SingleObjectiveGraphSpace(_SingleObjectiveSpace):
    """Non-tensorized single-objective graph space."""

    def __init__(
        self,
        n_agents: int,
        n_nodes: int,
        n_objectives: int,
        directed: bool = False,
        edge_prob: float = 0.3,
        connected: bool = False,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _SingleObjectiveSpace -> _SingleObjectiveGraphSpace."
        )

        self.n_nodes = n_nodes
        self.directed = directed
        self.edge_prob = edge_prob
        self.connected = connected

        super().__init__(n_agents, n_objectives, mapping, env)
        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.position = Graph.random(
                self.n_nodes, self.edge_prob, self.directed, self.connected
            )

        self.best_agent.position = self.agents[0].position.copy()
        self.best_agent.fit = copy.deepcopy(self.agents[0].fit)


class _MultiObjectiveGraphSpace(_MultiObjectiveSpace):
    """Non-tensorized multi-objective graph space."""

    def __init__(
        self,
        n_agents: int,
        n_nodes: int,
        n_objectives: int,
        directed: bool = False,
        edge_prob: float = 0.3,
        connected: bool = False,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _MultiObjectiveSpace -> _MultiObjectiveGraphSpace."
        )

        self.n_nodes = n_nodes
        self.directed = directed
        self.edge_prob = edge_prob
        self.connected = connected

        super().__init__(n_agents, n_objectives, mapping, env)
        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.position = Graph.random(
                self.n_nodes, self.edge_prob, self.directed, self.connected
            )


class _SingleObjectiveTensorGraphSpace(_SingleObjectiveTensorSpace):
    ...


class _MultiObjectiveTensorGraphSpace(_MultiObjectiveTensorSpace):
    ...


class GraphSpace:
    """A GraphSpace Factory Class for agents, variables and methods related
    to graph-structured search spaces (directed/undirected,
    tensorized/non-tensorized).
    """

    def __new__(
        cls,
        n_agents: int,
        n_nodes: int,
        n_objectives: int,
        directed: bool = False,
        edge_prob: float = 0.3,
        connected: bool = False,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
        tensorized: bool = False,
    ) -> Union[
        _SingleObjectiveGraphSpace,
        _MultiObjectiveGraphSpace,
        _SingleObjectiveTensorGraphSpace,
        _MultiObjectiveTensorGraphSpace,
    ]:
        """Initialization method.

        Args:
            n_agents: Number of space agents.
            n_nodes: Number of nodes per graph.
            n_objectives: Number of objective functions.
            directed: Whether generated/represented graphs are directed.
            edge_prob: Probability of an edge existing between any two
                nodes at initialization time.
            connected: Whether the graphs are connected.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.
            tensorized: Whether agents should be dense adjacency-matrix
                tensors (GPU/vectorization-friendly, fixed `n_nodes`) or
                arbitrary `Graph` objects (flexible topology).

        """
        if env is None:
            env = Environment("numpy", "float32")

        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be a positive integer.")
        if n_nodes <= 0:
            raise e.ValueError("`n_nodes` should be a positive integer.")

        if tensorized:
            raise e.Error(
                cls="GraphSpace",
                msg=("Tensorized GraphSpace is not implemented yet -- graphs "
                "currently only support the object-based representation.")
            )

        if n_objectives == 1:
            return _SingleObjectiveGraphSpace(
                n_agents,
                n_nodes,
                n_objectives,
                directed,
                edge_prob,
                connected,
                mapping,
                env,
            )
        else:
            return _MultiObjectiveGraphSpace(
                n_agents,
                n_nodes,
                n_objectives,
                directed,
                edge_prob,
                connected,
                mapping,
                env,
            )
