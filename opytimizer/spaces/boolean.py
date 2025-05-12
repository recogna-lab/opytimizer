"""Boolean-based search space.
"""

import copy
from typing import List, Optional

import numpy as np

from opytimizer.core import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class BooleanSpace(Space):
    """A BooleanSpace class for agents, variables and methods
    related to the boolean search space.

    """

    def __init__(
        self, n_agents: int, n_variables: int, mapping: Optional[List[str]] = None
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            mapping: String-based identifiers for mapping variables' names.

        """

        logger.info("Overriding class: Space -> BooleanSpace.")

        n_dimensions = 1
        lower_bound = np.zeros(n_variables)
        upper_bound = np.ones(n_variables)

        super(BooleanSpace, self).__init__(
            n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""

        for agent in self.agents:
            agent.fill_with_binary()

        self.best_agent = copy.deepcopy(self.agents[0])
