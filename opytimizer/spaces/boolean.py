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


class _SingleObjectiveBooleanSpace(_SingleObjectiveSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _SingleObjectiveSpace -> _SingleObjectiveBooleanSpace."
        )

        n_dimensions = 1
        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)
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
        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.fill_with_binary()
        self.best_agent = copy.deepcopy(self.agents[0])


class _MultiObjectiveBooleanSpace(_MultiObjectiveSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _MultiObjectiveSpace -> _MultiObjectiveBooleanSpace."
        )

        n_dimensions = 1
        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)
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
        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.fill_with_binary()


class _SingleObjectiveTensorBooleanSpace(_SingleObjectiveTensorSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:

        logger.info(
            "Overriding class: _SingleObjectiveTensorSpace -> _SingleObjectiveTensorBooleanSpace."
        )

        n_dimensions = 1
        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)
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
        self.build()

    def _initialize_agents(self) -> None:
        self.X[:] = self.env.xp.random.choice(
            [0, 1], size=(self.n_agents, self.n_variables, self.n_dimensions)
        ).astype(self.env.dtype)


class _MultiObjectiveTensorBooleanSpace(_MultiObjectiveTensorSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _MultiObjectiveTensorSpace -> _MultiObjectiveTensorBooleanSpace."
        )

        n_dimensions = 1
        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)
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
        self.build()

    def _initialize_agents(self) -> None:
        self.X[:] = self.env.xp.random.choice(
            [0, 1], size=(self.n_agents, self.n_variables, self.n_dimensions)
        ).astype(self.env.dtype)


class BooleanSpace:
    """
    A BooleanSpace Factory Class for agents, variables and methods
    related to the boolean search space.
    """

    def __new__(
        cls,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = None,
        tensorized: bool = False,
    ) -> Union[
        _SingleObjectiveBooleanSpace,
        _MultiObjectiveBooleanSpace,
        _SingleObjectiveTensorBooleanSpace,
        _MultiObjectiveTensorBooleanSpace,
    ]:
        """
        Initialization method.

            Args:
                n_agents: Number of space agents.
                n_variables: Number of decision variables.
                n_objectives: Number of objective functions.
                mapping: String-based identifiers for mapping variables' names.
                env: Environment class object.
                tensorized: Boolean flag that indicates a tensorized population usage

        """
        if env is None:
            env = Environment("numpy", "float32")

        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be a positive integer.")

        if tensorized:
            if n_objectives == 1:
                return _SingleObjectiveTensorBooleanSpace(
                    n_agents, n_variables, n_objectives, mapping, env
                )
            else:
                return _MultiObjectiveTensorBooleanSpace(
                    n_agents, n_variables, n_objectives, mapping, env
                )
        else:
            if n_objectives == 1:
                return _SingleObjectiveBooleanSpace(
                    n_agents, n_variables, n_objectives, mapping, env
                )
            else:
                return _MultiObjectiveBooleanSpace(
                    n_agents, n_variables, n_objectives, mapping, env
                )
