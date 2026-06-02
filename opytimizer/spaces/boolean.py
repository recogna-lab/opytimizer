"""Traditional-based search space.
"""

import copy
from typing import List, Optional, Tuple, Union


from opytimizer.core import _SingleObjectiveSpace, _MultiObjectiveSpace, Environment
from opytimizer.utils import logging


import opytimizer.utils.exception as e

logger = logging.get_logger(__name__)

class _SingleObjectiveBooleanSpace(_SingleObjectiveSpace):
    """A Single-Objective BooleanSpace class for agents, variables and methods
    related to the boolean space.

    """
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = Environment().set_backend('cpu')
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_objectives: Number of objective functions.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.
        """

        logger.info("Overriding class: _SingleObjectiveSpace -> _SingleObjectiveBooleanSpace.")

        n_dimensions = 1

        lower_bound = env.xp.zeros(n_variables, dtype=self.env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=self.env.dtype)

        super(_SingleObjectiveBooleanSpace, self).__init__(
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
            agent.fill_with_binary()

        self.best_agent = copy.deepcopy(self.agents[0])
        

class _MultiObjectiveBooleanSpace(_MultiObjectiveSpace):
    """A Multi-Objective BooleanSpace class for agents, variables and methods
    related to the boolean space.

    """
    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = Environment().set_backend('cpu')
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

        logger.info("Overriding class: _MultiObjectiveSpace -> _MultiObjectiveBooleanSpace.")

        n_dimensions = 1

        lower_bound = env.xp.zeros(n_variables, dtype=env.dtype)
        upper_bound = env.xp.ones(n_variables, dtype=env.dtype)

        super(_MultiObjectiveBooleanSpace, self).__init__(
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
            agent.fill_with_binary()

class BooleanSpace:
    """A Boolean Factory Space Class"""

    def __new__(
        self,
        n_agents: int,
        n_variables: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = Environment().set_backend('cpu')
    ) -> Union[_SingleObjectiveBooleanSpace, _MultiObjectiveBooleanSpace]:
        
        if n_objectives <= 0: raise e.ValueError('`n_objectives` should be a positive integer.')
        elif n_objectives == 1: return _SingleObjectiveBooleanSpace(n_agents, n_variables, n_objectives, mapping, env)
        else: return _MultiObjectiveBooleanSpace(n_agents, n_variables, n_objectives, mapping, env)