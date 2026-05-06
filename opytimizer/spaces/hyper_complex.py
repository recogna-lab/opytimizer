"""Hypercomplex-based search space.
"""

import copy
from typing import List, Optional, Union


from opytimizer.core import _MultiObjectiveSpace, _SingleObjectiveSpace, Environment
from opytimizer.utils import logging
import opytimizer.utils.exception as e

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
        env: Environment = Environment().set_backend('cpu')
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

        logger.info("Overriding class: _SingleObjectiveSpace -> _SingleObjectiveHyperComplexSpace.")

        lower_bound = self.env.xp.zeros(n_variables, dtype=self.env.dtype)
        upper_bound = self.env.xp.ones(n_variables, dtype=self.env.dtype)

        super(_SingleObjectiveHyperComplexSpace, self).__init__(
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
        env: Environment = Environment().set_backend('cpu')
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

        logger.info("Overriding class: _MultiObjectiveSpace -> _MultiObjectiveHyperComplexSpace.")

        lower_bound = self.env.xp.zeros(n_variables, dtype=self.env.dtype)
        upper_bound = self.env.xp.ones(n_variables, self.env.dtype)

        super(_MultiObjectiveHyperComplexSpace, self).__init__(
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
        """Initializes agents with their positions."""

        for agent in self.agents:
            agent.fill_with_uniform()


class HyperComplexSpace:
    """An HyperComplexSpace Factory Class that will hold agents, variables and methods
    related to the hypercomplex search space.

    """

    def __new__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        n_objectives: int,
        mapping: Optional[List[str]] = None,
        env: Environment = Environment().set_backend('cpu')
    ) -> Union[_SingleObjectiveHyperComplexSpace, _MultiObjectiveHyperComplexSpace]:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.
            n_objectives: Number of objective functions.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        if n_objectives <= 0: raise e.ValueError('`n_objectives` should be a positive integer.')
        elif n_objectives == 1: return _SingleObjectiveHyperComplexSpace(n_agents, n_variables, n_dimensions, n_objectives, mapping, env)
        else: return _MultiObjectiveHyperComplexSpace(n_agents, n_variables, n_dimensions, n_objectives, mapping, env)
        

        