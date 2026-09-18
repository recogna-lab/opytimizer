"""GraphAgent — an Agent whose position is a structured payload (a `Graph`
or `Tree`) instead of a numeric ndarray.
"""

import time
from typing import List, Optional, Union

import opytimizer.utils.exception as e
from opytimizer.core.environment import Environment
from opytimizer.core.graph.graph import Graph
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GraphAgent:
    """An Agent whose position is a `Graph` (or `Tree`, since `Tree`
    subclasses `Graph`) instance instead of a numeric ndarray.

    Args:
        n_objectives: Number of objective functions.
        mapping: Optional display name(s) for this agent's payload.
        env: Environment class object.

    """

    def __init__(
        self,
        n_objectives: int = 1,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        if env is None:
            env = Environment("numpy", "float32")

        self.env = env
        self.xp = env.xp

        self.n_objectives = n_objectives
        self._position: Optional[Graph] = None

        self._fit = self.xp.full(
            (n_objectives,), self.xp.finfo(self.xp.float64).max, dtype=self.xp.float64
        ).squeeze()

        self.mapping = mapping or ["graph"]
        self.ts = int(time.time())

    def __repr__(self) -> str:
        return f"GraphAgent(fit={self._fit}, position={self._position!r})"

    @property
    def n_objectives(self) -> int:
        """Number of objective functions."""

        return self._n_objectives

    @n_objectives.setter
    def n_objectives(self, n_objectives: int) -> None:
        if not isinstance(n_objectives, int):
            raise e.TypeError("`n_objectives` should be an integer")
        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be > 0")

        self._n_objectives = n_objectives

    @property
    def position(self) -> Graph:
        """The agent's payload: a `Graph` or `Tree` instance."""

        return self._position

    @position.setter
    def position(self, position: Union[Graph, "GraphAgent"]) -> None:
        if not isinstance(position, Graph):
            raise e.TypeError("`position` should be a `Graph` (or `Tree`) instance")

        self._position = position

    @property
    def fit(self):
        """Fitness value(s)."""

        return self._fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = fit

    @property
    def ts(self) -> int:
        """Timestamp of the agent."""

        return self._ts

    @ts.setter
    def ts(self, ts: int) -> None:
        if not isinstance(ts, int):
            raise e.TypeError("`ts` should be an integer")

        self._ts = ts

    @property
    def mapping(self) -> List[str]:
        """Payload mapping."""

        return self._mapping

    @mapping.setter
    def mapping(self, mapping: List[str]) -> None:
        self._mapping = mapping

    @property
    def env(self) -> Environment:
        return self._env

    @env.setter
    def env(self, env_instance) -> None:
        if not isinstance(env_instance, Environment):
            raise e.TypeError("Error: please, pass a valid environment.")

        self._env = env_instance

    def dominates(self, other: "GraphAgent") -> bool:
        """Checks if this agent dominates another agent.

        Args:
            other: Another agent to be compared.

        Returns:
            (bool): Whether this agent dominates the other.

        """

        return self.xp.all(self._fit <= other._fit) and self.xp.any(
            self._fit < other._fit
        )
