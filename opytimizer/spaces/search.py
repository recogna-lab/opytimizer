"""Traditional-based search space.
"""

import copy
from typing import List, Optional, Tuple, Union, Any

from opytimizer.core import _SingleObjectiveSpace, _MultiObjectiveSpace, Environment
from opytimizer.utils import logging

import opytimizer.utils.exception as e

logger = logging.get_logger(__name__)

class _SingleObjectiveSearchSpace(_SingleObjectiveSpace):
    """A Single-Objective SearchSpace class for agents, variables and methods
    related to the search space.

    """
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_objectives: Number of objective functions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.
        """

        logger.info("Overriding class: _SingleObjectiveSpace -> _SingleObjectiveSearchSpace.")

        n_dimensions = 1

        super(_SingleObjectiveSearchSpace, self).__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            n_objectives=n_objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            env=env
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""
        for agent in self.agents:
            agent.fill_with_uniform()

        self.best_agent = copy.deepcopy(self.agents[0])
        

class _MultiObjectiveSearchSpace(_MultiObjectiveSpace):
    """A Multi-Objective SearchSpace class for agents, variables and methods
    related to the search space.

    """
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_objectives: Number of objective functions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.
        """

        logger.info("Overriding class: _MultiObjectiveSpace -> _MultiObjectiveSearchSpace.")

        n_dimensions = 1

        super(_MultiObjectiveSearchSpace, self).__init__(
            n_agents=n_agents,
            n_variables=n_variables,
            n_dimensions=n_dimensions,
            n_objectives=n_objectives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mapping=mapping,
            env=env
        )

        self.build()

        logger.info("Class overrided.")


    def _initialize_agents(self) -> None:
        """Initializes agents with their positions.."""
        for agent in self.agents:
            agent.fill_with_uniform()

class SearchSpace:
    """Factory Search Space Class"""

    def __new__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None
    ) -> Union[_SingleObjectiveSearchSpace, _MultiObjectiveSearchSpace]:
        if env is None: env = Environment('cpu', 'float32')

        if n_objectives <= 0: raise e.ValueError('`n_objectives` should be a positive integer.')
        elif n_objectives == 1: return _SingleObjectiveSearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound, mapping, env)
        else: return _MultiObjectiveSearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound, mapping, env)