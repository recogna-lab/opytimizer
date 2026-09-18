"""Hypercomplex-based search space.
"""

import copy
from typing import List, Optional, Union

import opytimizer.utils.exception as e
from opytimizer.core import Environment
from opytimizer.core.space import (
    _MultiObjectiveSpace,
    _MultiObjectiveTensorSpace,
    _SingleObjectiveSpace,
    _SingleObjectiveTensorSpace,
)
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class _SingleObjectiveHyperComplexSpace(_SingleObjectiveSpace):
    """A SingleObjective HyperComplexSpace class that will hold agents, variables and methods
    related to the hypercomplex search space.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.
            n_objectives: Number of objective functions.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        logger.info(
            "Overriding class: _SingleObjectiveSpace -> _SingleObjectiveHyperComplexSpace."
        )

        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            n_objectives=n_objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            env=env,
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""

        for agent in self.agents:
            agent.fill_with_uniform()

        self.best_agent = copy.deepcopy(self.agents[0])


class _MultiObjectiveHyperComplexSpace(_MultiObjectiveSpace):
    """A MultiObjective HyperComplexSpace class that will hold agents, variables and methods
    related to the hypercomplex search space.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.
            n_objectives: Number of objective functions.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        logger.info(
            "Overriding class: _MultiObjectiveSpace -> _MultiObjectiveHyperComplexSpace."
        )

        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            n_objectives=n_objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            env=env,
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions."""

        for agent in self.agents:
            agent.fill_with_uniform()


class _SingleObjectiveTensorHyperComplexSpace(_SingleObjectiveTensorSpace):
    """A SingleObjective TensorHyperComplexSpace class that will hold agents, variables and methods
    related to the tensorized hypercomplex search space.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.
            n_objectives: Number of objective functions.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        logger.info(
            "Overriding class: _SingleObjectiveTensorSpace -> _SingleObjectiveTensorHyperComplexSpace."
        )

        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            n_objectives=n_objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            env=env,
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self) -> None:
        """Initializes agents' positions using a tensorized uniform distribution."""

        lb = self.lb.reshape(1, self.n_variables, self.n_dimensions)
        ub = self.ub.reshape(1, self.n_variables, self.n_dimensions)
        self.X[:] = self.env.xp.random.uniform(
            lb, ub, (self.n_agents, self.n_variables, self.n_dimensions)
        ).astype(self.env.dtype)


class _MultiObjectiveTensorHyperComplexSpace(_MultiObjectiveTensorSpace):
    """A MultiObjective TensorHyperComplexSpace class that will hold agents, variables and methods
    related to the tensorized hypercomplex search space.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.
            n_objectives: Number of objective functions.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        logger.info(
            "Overriding class: _MultiObjectiveTensorSpace -> _MultiObjectiveTensorHyperComplexSpace."
        )

        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)

        super().__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            n_objectives=n_objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            env=env,
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self) -> None:
        """Initializes agents' positions using a tensorized uniform distribution."""

        lb = self.lb.reshape(1, self.n_variables, self.n_dimensions)
        ub = self.ub.reshape(1, self.n_variables, self.n_dimensions)
        self.X[:] = self.env.xp.random.uniform(
            lb, ub, (self.n_agents, self.n_variables, self.n_dimensions)
        ).astype(self.env.dtype)


class HyperComplexSpace:
    """An HyperComplexSpace Factory Class that will hold agents, variables and methods
    related to the hypercomplex search space.

    """

    def __new__(
        cls,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
        tensorized: bool = False,
    ) -> Union[
        _SingleObjectiveHyperComplexSpace,
        _MultiObjectiveHyperComplexSpace,
        _SingleObjectiveTensorHyperComplexSpace,
        _MultiObjectiveTensorHyperComplexSpace,
    ]:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.
            n_objectives: Number of objective functions.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.
            tensorized: Boolean flag that indicates a tensorized population usage.

        """
        if env is None:
            env = Environment("numpy", "float32")

        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be a positive integer.")

        if tensorized:
            if n_objectives == 1:
                return _SingleObjectiveTensorHyperComplexSpace(
                    n_agents, n_variables, n_dimensions, n_objectives, mapping, env
                )
            else:
                return _MultiObjectiveTensorHyperComplexSpace(
                    n_agents, n_variables, n_dimensions, n_objectives, mapping, env
                )
        else:
            if n_objectives == 1:
                return _SingleObjectiveHyperComplexSpace(
                    n_agents, n_variables, n_dimensions, n_objectives, mapping, env
                )
            else:
                return _MultiObjectiveHyperComplexSpace(
                    n_agents, n_variables, n_dimensions, n_objectives, mapping, env
                )
