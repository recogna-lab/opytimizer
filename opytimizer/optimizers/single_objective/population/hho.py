"""Harris Hawks Optimization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class HHO(Optimizer):
    """An HHO class, inherited from Optimizer.

    This is the designed class to define HHO-related
    variables and methods.

    References:
        A. Heidari et al. Harris hawks optimization: Algorithm and applications.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> HHO.")

        super(HHO, self).__init__()

        self.build(params)

        logger.info("Class overrided.")

    def _calculate_initial_coefficients(
        self, iteration: int, n_iterations: int
    ) -> Tuple[float, float]:
        """Calculates the initial coefficients, i.e., energy and jump's strength.

        Args:
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        Returns:
            (Tuple[float, float]): Absolute value of energy and jump's strength.

        """

        r1 = r.generate_uniform_random_number()

        E_0 = 2 * r1 - 1
        J = 2 * (1 - r1)

        # Calculates the energy (eq. 3)
        E = 2 * E_0 * (1 - (iteration / n_iterations))

        return np.fabs(E), J

    def _exploration_phase(
        self, agents: List[Agent], current_agent: Agent, best_agent: Agent
    ) -> np.ndarray:
        """Performs the exploration phase.

        Args:
            agents: List of agents.
            current_agent: Current agent to be updated (or not).
            best_agent: Best population's agent.

        Returns:
            (np.ndarray): A location vector containing the updated position.

        """

        q = r.generate_uniform_random_number()
        if q >= 0.5:
            j = r.generate_integer_random_number(0, len(agents))

            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates the location vector (eq. 1 - part 1)
            location_vector = agents[j].position - r1 * np.fabs(
                agents[j].position - 2 * r2 * current_agent.position
            )
        else:
            average = np.mean([agent.position for agent in agents], axis=0)

            r3 = r.generate_uniform_random_number()
            r4 = r.generate_uniform_random_number()

            lb = np.expand_dims(current_agent.lb, -1)
            ub = np.expand_dims(current_agent.ub, -1)

            # Updates the location vector (eq. 1 - part 2)
            location_vector = (best_agent.position - average) - r3 * (
                lb + r4 * (ub - lb)
            )

        return location_vector

    def _exploitation_phase(
        self,
        energy: float,
        jump: float,
        agents: List[Agent],
        current_agent: Agent,
        best_agent: Agent,
        function: Function,
    ) -> np.ndarray:
        """Performs the exploitation phase.

        Args:
            energy: Energy coefficient.
            jump: Jump's strength.
            agents: List of agents.
            current_agent: Current agent to be updated (or not).
            best_agent: Best population's agent.
            function: A function object.

        Returns:
            (np.ndarray): A location vector containing the updated position.

        """

        w = r.generate_uniform_random_number()
        if w >= 0.5:
            # Soft besiege
            if energy >= 0.5:
                delta = best_agent.position - current_agent.position

                # Calculates the location vector (eq. 4)
                location_vector = delta - energy * np.fabs(
                    jump * best_agent.position - current_agent.position
                )

                return location_vector

            # Hard besiege
            else:
                delta = best_agent.position - current_agent.position

                # Calculates the location vector (eq. 6)
                location_vector = best_agent.position - energy * np.fabs(delta)

                return location_vector

        # With rapid dives
        # Soft besiege
        if energy >= 0.5:
            # Calculates the `Y` position (eq. 7)
            Y = best_agent.position - energy * np.fabs(
                jump * best_agent.position - current_agent.position
            )

            # Generates the Lévy's flight and random array (eq. 9)
            LF = d.generate_levy_distribution(
                1.5, (current_agent.n_variables, current_agent.n_dimensions)
            )
            S = r.generate_uniform_random_number(
                size=(current_agent.n_variables, current_agent.n_dimensions)
            )

            # Calculates the `Z` position (eq. 8)
            Z = Y + S * LF

            Y_fit = function(Y)
            Z_fit = function(Z)

            # If `Y` position is better than current agent's one (eq. 10 - part 1)
            if Y_fit < current_agent.fit:
                return Y

            # If `Z` position is better than current agent's one (eq. 10 - part 2)
            if Z_fit < current_agent.fit:
                return Z

        # Hard besiege
        else:
            average = np.mean([x.position for x in agents], axis=0)

            # Calculates the `Y` position (eq. 12)
            Y = best_agent.position - energy * np.fabs(
                jump * best_agent.position - average
            )

            # Generates the Lévy's flight and random array (eq. 9)
            LF = d.generate_levy_distribution(
                1.5, (current_agent.n_variables, current_agent.n_dimensions)
            )
            S = r.generate_uniform_random_number(
                size=(current_agent.n_variables, current_agent.n_dimensions)
            )

            # Calculates the `Z` position (eq. 13)
            Z = Y + S * LF

            Y_fit = function(Y)
            Z_fit = function(Z)

            # If `Y` position is better than current agent's one (eq. 11 - part 1)
            if Y_fit < current_agent.fit:
                return Y

            # If `Z` position is better than current agent's one (eq. 11 - part 2)
            if Z_fit < current_agent.fit:
                return Z

        return current_agent.position

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Harris Hawks Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        for agent in space.agents:
            E, J = self._calculate_initial_coefficients(iteration, n_iterations)
            if E >= 1:
                agent.position = self._exploration_phase(
                    space.agents, agent, space.best_agent
                )
            else:
                agent.position = self._exploitation_phase(
                    E, J, space.agents, agent, space.best_agent, function
                )
