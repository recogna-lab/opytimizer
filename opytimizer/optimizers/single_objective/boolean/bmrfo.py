"""Boolean Manta Ray Foraging Optimization.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class BMRFO(Optimizer):
    """A BMRFO class, inherited from Optimizer.

    This is the designed class to define boolean MRFO-related
    variables and methods.

    References:
        Publication pending.

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> BMRFO.")

        super(BMRFO, self).__init__()

        self.S = np.array([1])

        self.build(params)

        logger.info("Class overrided.")

    @property
    def S(self) -> np.ndarray:
        """Somersault foraging."""

        return self._S

    @S.setter
    def S(self, S: np.ndarray) -> None:
        if not isinstance(S, np.ndarray):
            raise e.TypeError("`S` should be a numpy array")

        self._S = S

    def _cyclone_foraging(
        self,
        agents: List[Agent],
        best_position: np.ndarray,
        i: int,
        iteration: int,
        n_iterations: int,
    ) -> np.ndarray:
        """Performs the cyclone foraging procedure.

        Args:
            agents: List of agents.
            best_position: Global best position.
            i: Current agent's index.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        Returns:
            (np.ndarray): A new cyclone foraging.

        """

        r1 = r.generate_binary_random_number(best_position.shape)
        beta = r.generate_binary_random_number(best_position.shape)

        u = r.generate_uniform_random_number()
        if iteration / n_iterations < u:
            r_position = r.generate_binary_random_number(
                size=(agents[i].n_variables, agents[i].n_dimensions)
            )

            if i == 0:
                partial_one = np.logical_or(
                    r1, np.logical_xor(r_position, agents[i].position)
                )
                partial_two = np.logical_or(
                    beta, np.logical_xor(r_position, agents[i].position)
                )
                cyclone_foraging = np.logical_and(
                    r_position, np.logical_and(partial_one, partial_two)
                )
            else:
                partial_one = np.logical_or(
                    r1, np.logical_xor(agents[i - 1].position, agents[i].position)
                )
                partial_two = np.logical_or(
                    beta, np.logical_xor(r_position, agents[i].position)
                )
                cyclone_foraging = np.logical_and(
                    r_position, np.logical_and(partial_one, partial_two)
                )
        else:
            if i == 0:
                partial_one = np.logical_or(
                    r1, np.logical_xor(best_position, agents[i].position)
                )
                partial_two = np.logical_or(
                    beta, np.logical_xor(best_position, agents[i].position)
                )
                cyclone_foraging = np.logical_and(
                    best_position, np.logical_and(partial_one, partial_two)
                )
            else:
                partial_one = np.logical_or(
                    r1, np.logical_xor(agents[i - 1].position, agents[i].position)
                )
                partial_two = np.logical_or(
                    beta, np.logical_xor(best_position, agents[i].position)
                )
                cyclone_foraging = np.logical_and(
                    best_position, np.logical_and(partial_one, partial_two)
                )

        return cyclone_foraging

    def _chain_foraging(
        self, agents: List[Agent], best_position: np.ndarray, i: int
    ) -> np.ndarray:
        """Performs the chain foraging procedure.

        Args:
            agents: List of agents.
            best_position: Global best position.
            i: Current agent's index.

        Returns:
            (np.ndarray): A new chain foraging.

        """

        r1 = r.generate_binary_random_number(best_position.shape)
        alpha = r.generate_binary_random_number(best_position.shape)

        if i == 0:
            partial_one = np.logical_and(
                r1, np.logical_xor(best_position, agents[i].position)
            )
            partial_two = np.logical_and(
                alpha, np.logical_xor(best_position, agents[i].position)
            )
            chain_foraging = np.logical_or(
                agents[i].position, np.logical_or(partial_one, partial_two)
            )
        else:
            partial_one = np.logical_and(
                r1, np.logical_xor(agents[i - 1].position, agents[i].position)
            )
            partial_two = np.logical_and(
                alpha, np.logical_xor(best_position, agents[i].position)
            )
            chain_foraging = np.logical_or(
                agents[i].position, np.logical_or(partial_one, partial_two)
            )

        return chain_foraging

    def _somersault_foraging(
        self, position: np.ndarray, best_position: np.ndarray
    ) -> np.ndarray:
        """Performs the somersault foraging procedure.

        Args:
            position: Agent's current position.
            best_position: Global best position.

        Returns:
            (np.ndarray): A new somersault foraging.

        """

        r1 = r.generate_binary_random_number(best_position.shape)
        r2 = r.generate_binary_random_number(best_position.shape)

        somersault_foraging = np.logical_or(
            position,
            np.logical_and(
                self.S,
                np.logical_xor(
                    np.logical_xor(r1, best_position), np.logical_xor(r2, position)
                ),
            ),
        )

        return somersault_foraging

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Boolean Manta Ray Foraging Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        for i, agent in enumerate(space.agents):
            r1 = r.generate_uniform_random_number()
            if r1 < 0.5:
                agent.position = self._cyclone_foraging(
                    space.agents, space.best_agent.position, i, iteration, n_iterations
                )
            else:
                agent.position = self._chain_foraging(
                    space.agents, space.best_agent.position, i
                )

            agent.clip_by_bound()

            agent.fit = function(agent.position)
            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)

        for agent in space.agents:
            agent.position = self._somersault_foraging(
                agent.position, space.best_agent.position
            )
