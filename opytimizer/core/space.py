from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Agent
from opytimizer.core.environment import Environment
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


def _cached_clip_bounds(space):
    """Returns (lb, ub) shaped (1, n_variables, n_dimensions) in the population dtype.

    `clip_by_bound` runs once per iteration. Bounds stored as float64 would make
    `clip(X_float32, lb_float64, ub_float64, out=X)` compute in float64 and cast
    back, so they are converted once here.
    """
    cache = getattr(space, "_clip_bounds_cache", None)
    if cache is None or cache[0] is not space.lb or cache[1] is not space.ub:
        xp = space.env.xp
        shape = (1, space.n_variables, space.n_dimensions)
        lb = xp.asarray(space.lb, dtype=space.env.dtype).reshape(shape)
        ub = xp.asarray(space.ub, dtype=space.env.dtype).reshape(shape)
        cache = (space.lb, space.ub, lb, ub)
        space._clip_bounds_cache = cache

    return cache[2], cache[3]


def _nondominated_mask(costs, xp, max_elements: int = 1 << 26):
    """Boolean mask of the rows of `costs` (n_agents, n_objectives) that are
    non-dominated and not duplicates of an earlier row (minimization).

    Equivalent to the dense N x N x M pairwise comparison, but the targets are
    processed in row blocks so every temporary holds at most ~`max_elements`
    booleans.
    """
    n_agents, n_objectives = costs.shape
    if n_agents == 0:
        return xp.zeros((0,), dtype=bool)

    chunk = max(1, max_elements // max(1, n_agents * n_objectives))
    index = xp.arange(n_agents)
    opponents = costs[xp.newaxis, :, :]
    blocks = []

    for start in range(0, n_agents, chunk):
        stop = min(start + chunk, n_agents)
        targets = costs[start:stop, xp.newaxis, :]

        # dominated[i]: some opponent j is no worse everywhere and better somewhere.
        no_worse = xp.all(opponents <= targets, axis=-1)
        better = xp.any(opponents < targets, axis=-1)
        is_dominated = xp.any(no_worse & better, axis=1)

        # duplicate[i]: an identical row with a LOWER index exists (keep the first).
        identical = xp.all(opponents == targets, axis=-1)
        earlier = index[xp.newaxis, :] < index[start:stop, xp.newaxis]
        is_duplicate = xp.any(identical & earlier, axis=1)

        blocks.append(~(is_dominated | is_duplicate))

    return blocks[0] if len(blocks) == 1 else xp.concatenate(blocks)


class _Space(ABC):
    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        n_objectives: int = 1,
        lower_bound: Optional[Union[float, List, Tuple, Any]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, Any]] = 1.0,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        self.n_agents = n_agents
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions
        self.n_objectives = n_objectives
        if env is None:
            env = Environment().set_backend("numpy")
        self.env = env
        self.lb = self.env.xp.asarray(lower_bound)
        self.ub = self.env.xp.asarray(upper_bound)
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
    def n_variables(self) -> int:
        return self._n_variables

    @n_variables.setter
    def n_variables(self, n_variables: int) -> None:
        if not isinstance(n_variables, int):
            raise e.TypeError("`n_variables` should be an integer")
        if n_variables <= 0:
            raise e.ValueError("`n_variables` should be > 0")
        self._n_variables = n_variables

    @property
    def n_dimensions(self) -> int:
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions: int) -> None:
        if not isinstance(n_dimensions, int):
            raise e.TypeError("`n_dimensions` should be an integer")
        if n_dimensions <= 0:
            raise e.ValueError("`n_dimensions` should be > 0")
        self._n_dimensions = n_dimensions

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
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, lb) -> None:
        if not lb.shape:
            lb = self.env.xp.expand_dims(lb, -1)
        if lb.shape[0] != self.n_variables:
            raise e.SizeError("`lb` should be the same size as `n_variables`")
        self._lb = lb

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, ub) -> None:
        if not ub.shape:
            ub = self.env.xp.expand_dims(ub, -1)
        if not ub.shape or ub.shape[0] != self.n_variables:
            raise e.SizeError("`ub` should be the same size as `n_variables`")
        self._ub = ub

    @property
    def mapping(self) -> List[str]:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: List[str]) -> None:
        if mapping is not None:
            if not isinstance(mapping, list):
                raise e.TypeError("`mapping` should be a list")
            if len(mapping) != self.n_variables:
                raise e.SizeError("`mapping` should be the same size as `n_variables`")
            self._mapping = mapping
        else:
            self._mapping = [f"x{i}" for i in range(self.n_variables)]

    @property
    def env(self) -> Environment:
        return self._env

    @env.setter
    def env(self, env_instance) -> None:
        if not isinstance(env_instance, Environment):
            raise e.TypeError("Error: please, pass a valiable environment.")
        self._env = env_instance

    @property
    def agents(self) -> List[Agent]:
        return self._agents

    @agents.setter
    def agents(self, agents: List[Agent]) -> None:
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
            "Agents: %d | Size: (%d, %d) | Built: %s.",
            self.n_agents,
            self.n_variables,
            self.n_dimensions,
            self.built,
        )

    def clip_by_bound(self) -> None:
        for agent in self.agents:
            agent.clip_by_bound()


class _SingleObjectiveSpace(_Space):
    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        n_objectives: int = 1,
        lower_bound: Optional[Union[float, List, Tuple, Any]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, Any]] = 1.0,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )
        self.best_agent = Agent(
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            n_objectives=n_objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            env=env,
        )

        self.built = False

    @property
    def best_agent(self) -> Agent:
        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent: Agent) -> None:
        if not isinstance(best_agent, Agent):
            raise e.TypeError("`best_agent` should be an Agent")
        self._best_agent = best_agent

    def _create_agents(self) -> None:
        self.agents = [
            Agent(
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                n_objectives=self.n_objectives,
                lower_bound=self.lb,
                upper_bound=self.ub,
                mapping=self.mapping,
                env=self.env,
            )
            for _ in range(self.n_agents)
        ]


class _MultiObjectiveSpace(_Space):
    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        n_objectives: int = 1,
        lower_bound: Optional[Union[float, List, Tuple, Any]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, Any]] = 1.0,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )

        self.pareto_front = []
        self.built = False

    @property
    def pareto_front(self) -> List[Agent]:
        return self._pareto_front

    @pareto_front.setter
    def pareto_front(self, pareto_front: Union[List[Agent], Any]) -> None:
        self._pareto_front = pareto_front

    def _create_agents(self) -> None:
        self.agents = [
            Agent(
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                n_objectives=self.n_objectives,
                lower_bound=self.lb,
                upper_bound=self.ub,
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
        valid_mask = _nondominated_mask(costs, xp)
        if hasattr(valid_mask, "get"):
            valid_mask = valid_mask.get()
        self.pareto_front = [self.agents[i] for i in range(n_agents) if valid_mask[i]]


class _SingleObjectiveTensorSpace(_SingleObjectiveSpace):
    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        n_objectives: int = 1,
        lower_bound: Optional[Union[float, List, Tuple, Any]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, Any]] = 1.0,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        self.X = None
        self.F = None
        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )

    def _create_agents(self) -> None:
        xp = self.env.xp

        _env = Environment("numpy", self.env.dtype)
        self.X = self.env.xp.zeros(
            (self.n_agents, self.n_variables, self.n_dimensions), dtype=self.env.dtype
        )
        self.F = self.env.xp.zeros((self.n_agents,), dtype=self.env.dtype)
        if xp.__name__ == "cupy":
            _lb = xp.asnumpy(self.lb)
            _ub = xp.asnumpy(self.ub)
        else:
            _lb = self.lb
            _ub = self.ub

        for _ in range(self.n_agents):
            agent = Agent(
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                n_objectives=self.n_objectives,
                lower_bound=_lb,
                upper_bound=_ub,
                mapping=self.mapping,
                env=_env,
            )

            self.agents.append(agent)

        self.best_agent = Agent(
            n_variables=self.n_variables,
            n_dimensions=self.n_dimensions,
            n_objectives=self.n_objectives,
            lower_bound=_lb,
            upper_bound=_ub,
            mapping=self.mapping,
            env=_env,
        )

    def clip_by_bound(self):
        # Bounds are cast/reshaped once and cached (see `_cached_clip_bounds`)
        # instead of being rebuilt on every iteration.
        lb, ub = _cached_clip_bounds(self)
        self.env.xp.clip(self.X, lb, ub, out=self.X)


class _MultiObjectiveTensorSpace(_MultiObjectiveSpace):
    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        n_objectives: int = 1,
        lower_bound: Optional[Union[float, List, Tuple, Any]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, Any]] = 1.0,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        self.X = None
        self.F = None
        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )

    def _create_agents(self) -> None:
        xp = self.env.xp

        _env = Environment("numpy", self.env.dtype)
        self.X = self.env.xp.zeros(
            (self.n_agents, self.n_variables, self.n_dimensions), dtype=self.env.dtype
        )
        self.F = self.env.xp.zeros(
            (self.n_agents, self.n_objectives), dtype=self.env.dtype
        )
        if xp.__name__ == "cupy":
            _lb = xp.asnumpy(self.lb)
            _ub = xp.asnumpy(self.ub)
        else:
            _lb = self.lb
            _ub = self.ub

        for _ in range(self.n_agents):
            agent = Agent(
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                n_objectives=self.n_objectives,
                lower_bound=_lb,
                upper_bound=_ub,
                mapping=self.mapping,
                env=_env,
            )

            self.agents.append(agent)

    def clip_by_bound(self):
        # Bounds are cast/reshaped once and cached (see `_cached_clip_bounds`)
        # instead of being rebuilt on every iteration.
        lb, ub = _cached_clip_bounds(self)
        self.env.xp.clip(self.X, lb, ub, out=self.X)

    def update_pareto_front(self, _xp=np) -> None:
        # `_xp` is kept only for backwards compatibility with existing callers:
        # X and F were allocated with `self.env.xp`.
        if self.X is None or self.F is None:
            self.pareto_front = []
            return

        valid_mask = _nondominated_mask(self.F, self.env.xp)

        # The mask stays on its original device.
        self.pareto_front = (self.X[valid_mask], self.F[valid_mask])
