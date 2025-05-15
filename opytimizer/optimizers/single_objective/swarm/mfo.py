"""Moth-Flame Optimization.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as rnd
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class MFO(Optimizer):
    """A MFO class, inherited from Optimizer.

    This is the designed class to define MFO-related
    variables and methods.

    References:
        S. Mirjalili. Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
        Knowledge-Based Systems (2015).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> MFO.")

        super(MFO, self).__init__()

        self.b = 1

        self.build(params)

        logger.info("Class overrided.")

    @property
    def b(self) -> float:
        """Spiral constant."""

        return self._b

    @b.setter
    def b(self, b: float) -> None:
        if not isinstance(b, (float, int)):
            raise e.TypeError("`b` should be a float or integer")
        if b < 0:
            raise e.ValueError("`b` should be >= 0")

        self._b = b

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Moth-Flame Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        flames = copy.deepcopy(space.agents)
        flames.sort(key=lambda x: x.fit)

        # Calculates the number of flames (eq. 3.14)
        n_flames = int(len(flames) - iteration * ((len(flames) - 1) / n_iterations)) - 1

        r = -1 + iteration * (-1 / n_iterations)

        for i, agent in enumerate(space.agents):
            for j in range(agent.n_variables):
                t = rnd.generate_uniform_random_number(r, 1)

                if i < n_flames:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[i].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = (
                        D * np.exp(self.b * t) * np.cos(2 * np.pi * t)
                        + flames[i].position[j]
                    )
                else:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[0].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = (
                        D * np.exp(self.b * t) * np.cos(2 * np.pi * t)
                        + flames[0].position[j]
                    )
