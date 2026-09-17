from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.environment import Environment
from opytimizer.core.graph import GraphAgent
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class _Space(ABC):
    def __init__(
        self,
        n_agents: int = 1,
        n_objectives: int = 1,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        self.n_agents = n_agents
        self.n_objectives = n_objectives
        if env is None:
            env = Environment().set_backend("numpy")
        self.env = env

        self.mapping = mapping
        self.agents = []

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @n_agents.setter
    def n_agents(self, n_agents: int) -> None:
        if not isinstance(n_agents, int):
            raise e.TypeError("`n_agents` should be an integer")
        if n_agents <= 0:
            raise e.ValueError("`n_agents` should be > 0")
        self._n_agents = n_agents

    @property
    def n_objectives(self) -> int:
        return self._n_objectives

    @n_objectives.setter
    def n_objectives(self, n_objectives: int) -> None:
        if not isinstance(n_objectives, int):
            raise e.TypeError("`n_objectives` should be an integer")
        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be > 0")
        self._n_objectives = n_objectives

    @property
    def env(self) -> Environment:
        return self._env

    @env.setter
    def env(self, env_instance) -> None:
        if not isinstance(env_instance, Environment):
            raise e.TypeError("Error: please, pass a valiable environment.")
        self._env = env_instance

    @property
    def agents(self) -> List[GraphAgent]:
        return self._agents

    @agents.setter
    def agents(self, agents: List[GraphAgent]) -> None:
        if not isinstance(agents, list):
            raise e.TypeError("`agents` should be a list")
        self._agents = agents

    @property
    def built(self) -> bool:
        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        if not isinstance(built, bool):
            raise e.TypeError("`built` should be a boolean")
        self._built = built

    @abstractmethod
    def _create_agents(self) -> None:
        pass

    def _initialize_agents(self) -> None:
        pass

    def build(self) -> None:
        self._create_agents()
        self._initialize_agents()
        self.built = True

        logger.debug(
            "Agents: %d | Built: %s.",
            self.n_agents,
            self.built,
        )

    def clip_by_bound(self) -> None:
        return


class _SingleObjectiveSpace(_Space):
    def __init__(
        self,
        n_agents: int = 1,
        n_objectives: int = 1,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        super().__init__(
            n_agents,
            n_objectives,
            mapping,
            env,
        )
        self.best_agent = GraphAgent(
            n_objectives=n_objectives,
            mapping=mapping,
            env=env,
        )

        self.built = False

    @property
    def best_agent(self) -> GraphAgent:
        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent: GraphAgent) -> None:
        if not isinstance(best_agent, GraphAgent):
            raise e.TypeError("`best_agent` should be a GraphAgent")
        self._best_agent = best_agent

    def _create_agents(self) -> None:
        self.agents = [
            GraphAgent(
                n_objectives=self.n_objectives,
                mapping=self.mapping,
                env=self.env,
            )
            for _ in range(self.n_agents)
        ]


class _MultiObjectiveSpace(_Space):
    def __init__(
        self,
        n_agents: int = 1,
        n_objectives: int = 1,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        super().__init__(
            n_agents,
            n_objectives,
            mapping,
            env,
        )

        self.pareto_front = []
        self.built = False

    @property
    def pareto_front(self) -> List[GraphAgent]:
        return self._pareto_front

    @pareto_front.setter
    def pareto_front(self, pareto_front: Union[List[GraphAgent], Any]) -> None:
        self._pareto_front = pareto_front

    def _create_agents(self) -> None:
        self.agents = [
            GraphAgent(
                n_objectives=self.n_objectives,
                mapping=self.mapping,
                env=self.env,
            )
            for _ in range(self.n_agents)
        ]

    def update_pareto_front(self, **kwargs) -> None:
        if not self.agents:
            self.pareto_front = []
            return
        xp = self.env.xp
        costs = xp.stack([xp.asarray(agent.fit).ravel() for agent in self.agents])
        n_agents = costs.shape[0]
        targets = costs[:, xp.newaxis, :]
        opponents = costs[xp.newaxis, :, :]
        no_worse = xp.all(opponents <= targets, axis=-1)
        better = xp.any(opponents < targets, axis=-1)
        dominates = no_worse & better
        is_dominated = xp.any(dominates, axis=1)
        identical = xp.all(opponents == targets, axis=-1)
        lower_tri = xp.tril(xp.ones((n_agents, n_agents), dtype=bool), k=-1)
        is_duplicate = xp.any(identical & lower_tri, axis=1)
        valid_mask = ~(is_dominated | is_duplicate)
        if hasattr(valid_mask, "get"):
            valid_mask = valid_mask.get()
        self.pareto_front = [self.agents[i] for i in range(n_agents) if valid_mask[i]]


class _SingleObjectiveTensorSpace(_SingleObjectiveSpace):
    def __init__(
        self,
        n_agents: int = 1,
        n_objectives: int = 1,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        self.F = None

        super().__init__(
            n_agents,
            n_objectives,
            mapping,
            env,
        )

    def _create_agents(self) -> None:

        _env = Environment("numpy", self.env.dtype)

        self.F = self.env.xp.zeros((self.n_agents,), dtype=self.env.dtype)

        for _ in range(self.n_agents):
            agent = GraphAgent(
                n_objectives=self.n_objectives,
                mapping=self.mapping,
                env=_env,
            )

            self.agents.append(agent)

        self.best_agent = GraphAgent(
            n_objectives=self.n_objectives,
            mapping=self.mapping,
            env=_env,
        )

    def clip_by_bound(self):
        return


class _MultiObjectiveTensorSpace(_MultiObjectiveSpace):
    def __init__(
        self,
        n_agents: int = 1,
        n_objectives: int = 1,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        self.F = None

        super().__init__(
            n_agents,
            n_objectives,
            mapping,
            env,
        )

    def _create_agents(self) -> None:

        _env = Environment("numpy", self.env.dtype)

        self.F = self.env.xp.zeros(
            (self.n_agents, self.n_objectives), dtype=self.env.dtype
        )

        for _ in range(self.n_agents):
            agent = GraphAgent(
                n_objectives=self.n_objectives,
                mapping=self.mapping,
                env=_env,
            )

            self.agents.append(agent)

    def clip_by_bound(self):
        return

    def update_pareto_front(self, _xp=np) -> None:
        if self.X is None or self.F is None:
            self.pareto_front = []
            return

        costs = self.F
        n_agents = costs.shape[0]
        targets = costs[:, _xp.newaxis, :]
        opponents = costs[_xp.newaxis, :, :]
        no_worse = _xp.all(opponents <= targets, axis=-1)
        better = _xp.any(opponents < targets, axis=-1)
        dominates = no_worse & better
        is_dominated = _xp.any(dominates, axis=1)
        identical = _xp.all(opponents == targets, axis=-1)
        lower_tri = _xp.tril(_xp.ones((n_agents, n_agents), dtype=bool), k=-1)
        is_duplicate = _xp.any(identical & lower_tri, axis=1)
        valid_mask = ~(is_dominated | is_duplicate)
        if hasattr(valid_mask, "get"):
            valid_mask = valid_mask.get()
        self.pareto_front = (self.X[valid_mask], self.F[valid_mask])
