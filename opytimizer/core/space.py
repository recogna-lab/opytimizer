"""Search space.
"""

from typing import List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

import opytimizer.utils.exception as e
from opytimizer.core import Agent
from opytimizer.utils import logging
from opytimizer.core.environment import Environment
logger = logging.get_logger(__name__)


class _Space(ABC):
    """A Space class for agents, variables and methods
    related to the search space.

    """

    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        n_objectives: int = 1,
        lower_bound: Optional[Union[float, List, Tuple, Any]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, Any]] = 1.0,
        mapping: Optional[List[str]] = None,
        env: Environment = Environment().set_backend('cpu')
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Dimension of search space.
            n_objectives: Number of objective functions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        self.n_agents = n_agents
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions
        self.n_objectives = n_objectives

        self.lb = env.xp.asarray(lower_bound)
        self.ub = env.xp.asarray(upper_bound)

        self.mapping = mapping

        self.env = env

        self.agents = []


    @property
    def n_agents(self) -> int:
        """Number of agents."""

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
        """Number of decision variables."""

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
        """Number of search space dimensions."""

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
    def lb(self):
        """Minimum possible values."""

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
        """Maximum possible values."""

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
        """Variables mapping."""

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
        """list: Agents that belongs to the space."""

        return self._agents

    @agents.setter
    def agents(self, agents: List[Agent]) -> None:
        if not isinstance(agents, list):
            raise e.TypeError("`agents` should be a list")

        self._agents = agents


    @property
    def built(self) -> bool:
        """Indicates whether the space is built."""

        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        if not isinstance(built, bool):
            raise e.TypeError("`built` should be a boolean")

        self._built = built

    @abstractmethod
    def _create_agents(self) -> None:
        """Creates a list of agents."""
        pass
    
    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent.

        As each child has a different procedure of initialization,
        you will need to implement it directly on its class.

        """

        pass

    def build(self) -> None:
        """Builds the object by creating and initializing the agents."""

        self._create_agents()
        self._initialize_agents()

        self.built = True

        logger.debug(
            "Agents: %d | Size: (%d, %d) | "
            "Objectives: %d | Lower Bound: %s | Upper Bound: %s | "
            "Mapping: %s | Built: %s.",
            self.n_agents,
            self.n_variables,
            self.n_dimensions,
            self.n_objectives,
            self.lb,
            self.ub,
            self.mapping,
            self.built,
        )

    def clip_by_bound(self) -> None:
        """Clips the agents' decision variables to the bounds limits."""

        for agent in self.agents:
            agent.clip_by_bound()


### SINGLE OBJECTIVE SPACE ---------------------------------------------------
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
        env: Environment = Environment().set_backend('cpu')
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Dimension of search space.
            n_objectives: Number of objective functions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Enviromennt class object.

        """
        super().__init__(n_agents, n_variables, n_dimensions, n_objectives, lower_bound, upper_bound, mapping, env)

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
        """Agent: Best agent."""

        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent: Agent) -> None:
        if not isinstance(best_agent, Agent):
            raise e.TypeError("`best_agent` should be an Agent")

        self._best_agent = best_agent


    def _create_agents(self) -> None:
        """Creates a list of agents."""

        self.agents = [
            Agent(
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                n_objectives=self.n_objectives,
                lower_bound=self.lb,
                upper_bound=self.ub,
                mapping=self.mapping,
                env=self.env
            )
            for _ in range(self.n_agents)
        ]



    
## --------MULTI-OBJECTIVE SPACE ---------------------------------

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
        env: Environment = Environment().set_backend('cpu')
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Dimension of search space.
            n_objectives: Number of objective functions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """
        super().__init__(n_agents, n_variables, n_dimensions, n_objectives, lower_bound, upper_bound, mapping, env)

        self.pareto_front = []

        self.built = False

    @property
    def pareto_front(self) -> List[Agent]:
        """List of non-dominated solutions."""
        return self._pareto_front

    @pareto_front.setter
    def pareto_front(self, pareto_front: List[Agent]) -> None:
        if not isinstance(pareto_front, list):
            raise e.TypeError("`pareto_front` should be a list")

        self._pareto_front = pareto_front

    def _create_agents(self) -> None:
        """Creates a list of agents."""

        self.agents = [
            Agent(
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                n_objectives=self.n_objectives,
                lower_bound=self.lb,
                upper_bound=self.ub,
                mapping=self.mapping,
                env=self.env
            )
            for _ in range(self.n_agents)
        ]

        self.best_agent = Agent(
            n_variables=self.n_variables,
            n_dimensions=self.n_dimensions,
            n_objectives=self.n_objectives,
            lower_bound=self.lb,
            upper_bound=self.ub,
            mapping=self.mapping,
            env=self.env
        )

    def update_pareto_front(self) -> None:
        """Updates the Pareto front with non-dominated solutions.

        Args:
            agents: List of agents to be evaluated.

        """
        self.pareto_front = []
        for agent in self.agents:
            is_dominated = False
            is_duplicate = any(
                self.env.xp.array_equal(agent.fit, existing_agent.fit)
                for existing_agent in self.pareto_front
            )
            if is_duplicate:
                continue
            for pareto_agent in self.pareto_front:
                if pareto_agent.dominates(agent):
                    is_dominated = True
                    break
            if not is_dominated:
                self.pareto_front = [
                    a for a in self.pareto_front if not agent.dominates(a)
                ]
                self.pareto_front.append(agent)