"""Typed tree search spaces (strongly-typed genetic programming).
"""

import copy
from typing import List, Optional

import opytimizer.utils.exception as e
from opytimizer.core import Environment
from opytimizer.core.graph.generator import generate_typed_tree
from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.core.graph.space import _MultiObjectiveSpace, _SingleObjectiveSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class _SingleObjectiveTreeSpace(_SingleObjectiveSpace):
    """Single-objective space whose agents are typed GP trees."""

    def __init__(
        self,
        n_agents: int,
        n_objectives: int,
        pset: PrimitiveSet,
        min_depth: int = 2,
        max_depth: int = 6,
        method: str = "half_and_half",
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _SingleObjectiveSpace -> _SingleObjectiveTreeSpace."
        )

        self.pset = pset
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.method = method

        super().__init__(n_agents, n_objectives, mapping, env)
        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.position = generate_typed_tree(
                self.pset, self.min_depth, self.max_depth, self.method
            )
        self.best_agent = copy.deepcopy(self.agents[0])


class _MultiObjectiveTreeSpace(_MultiObjectiveSpace):
    """Multi-objective space whose agents are typed GP trees."""

    def __init__(
        self,
        n_agents: int,
        n_objectives: int,
        pset: PrimitiveSet,
        min_depth: int = 2,
        max_depth: int = 6,
        method: str = "half_and_half",
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _MultiObjectiveSpace -> _MultiObjectiveTreeSpace."
        )

        self.pset = pset
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.method = method

        super().__init__(n_agents, n_objectives, mapping, env)
        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.position = generate_typed_tree(
                self.pset, self.min_depth, self.max_depth, self.method
            )


class TreeSpace:
    """A TreeSpace Factory Class for agents, variables and methods related
    to strongly-typed genetic programming trees.

    Note:
        A `tensorized` variant is a natural future extension point but is not implemented
        yet, since tree topologies are inherently variable-sized.
        Passing `tensorized=True` raises `NotImplementedError` rather than
        failing silently.

    """

    def __new__(
        cls,
        n_agents: int,
        n_objectives: int,
        pset: PrimitiveSet,
        min_depth: int = 2,
        max_depth: int = 6,
        method: str = "half_and_half",
        mapping: Optional[List[str]] = None,
        env: Environment = None,
        tensorized: bool = False,
    ):
        """Initialization method.

        Args:
            n_agents: Number of space agents.
            n_objectives: Number of objective functions.
            pset: Registry of typed primitives/terminals used to grow
                trees (see `opytimizer.core.graph.primitive_set.PrimitiveSet`).
            min_depth: Minimum tree depth at initialization.
            max_depth: Maximum tree depth at initialization.
            method: One of `grow`, `full` or `half_and_half`.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.
            tensorized: Reserved for a future dense/fixed-shape tree
                encoding. Currently unsupported.

        """
        if env is None:
            env = Environment("numpy", "float32")

        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be a positive integer.")
        if tensorized:
            raise e.Error(
                cls="TreeSpace",
                msg=(
                    "Tensorized TreeSpace is not implemented yet -- typed trees "
                    "currently only support the object-based representation."
                ),
            )

        if n_objectives == 1:
            return _SingleObjectiveTreeSpace(
                n_agents, n_objectives, pset, min_depth, max_depth, method, mapping, env
            )
        else:
            return _MultiObjectiveTreeSpace(
                n_agents, n_objectives, pset, min_depth, max_depth, method, mapping, env
            )
