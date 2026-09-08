import copy
from typing import Any, List, Optional, Tuple, Union

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


class _SingleObjectiveSearchSpace(_SingleObjectiveSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _SingleObjectiveSpace -> _SingleObjectiveSearchSpace."
        )

        n_dimensions = 1
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
            agent.fill_with_uniform()
        self.best_agent = copy.deepcopy(self.agents[0])


class _MultiObjectiveSearchSpace(_MultiObjectiveSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _MultiObjectiveSpace -> _MultiObjectiveSearchSpace."
        )

        n_dimensions = 1
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
            agent.fill_with_uniform()


class _SingleObjectiveTensorSearchSpace(_SingleObjectiveTensorSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _SingleObjectiveTensorSpace -> _SingleObjectiveTensorSearchSpace."
        )

        n_dimensions = 1
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
        lb = self.lb.reshape(1, self.n_variables, self.n_dimensions)
        ub = self.ub.reshape(1, self.n_variables, self.n_dimensions)
        self.X[:] = self.env.xp.random.uniform(
            lb, ub, (self.n_agents, self.n_variables, self.n_dimensions)
        ).astype(self.env.dtype)


class _MultiObjectiveTensorSearchSpace(_MultiObjectiveTensorSpace):
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _MultiObjectiveTensorSpace -> _MultiObjectiveTensorSearchSpace."
        )

        n_dimensions = 1
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
        lb = self.lb.reshape(1, self.n_variables, self.n_dimensions)
        ub = self.ub.reshape(1, self.n_variables, self.n_dimensions)
        self.X[:] = self.env.xp.random.uniform(
            lb, ub, (self.n_agents, self.n_variables, self.n_dimensions)
        ).astype(self.env.dtype)


class SearchSpace:
    """
    A SearchSpace Factory Class for agents, variables and methods
    related to the search space.
    """

    def __new__(
        cls,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
        tensorized: bool = False,
    ) -> Union[
        _SingleObjectiveSearchSpace,
        _MultiObjectiveSearchSpace,
        _SingleObjectiveTensorSearchSpace,
        _MultiObjectiveTensorSearchSpace,
    ]:
        """
        Initialization method.

                Args:
                    n_agents: Number of space agents.
                    n_variables: Number of decision variables.
                    n_objectives: Number of objective functions.
                    lower_bound: Minimum possible values.
                    upper_bound: Maximum possible values.
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
                return _SingleObjectiveTensorSearchSpace(
                    n_agents,
                    n_variables,
                    n_objectives,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
            else:
                return _MultiObjectiveTensorSearchSpace(
                    n_agents,
                    n_variables,
                    n_objectives,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
        else:
            if n_objectives == 1:
                return _SingleObjectiveSearchSpace(
                    n_agents,
                    n_variables,
                    n_objectives,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
            else:
                return _MultiObjectiveSearchSpace(
                    n_agents,
                    n_variables,
                    n_objectives,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
