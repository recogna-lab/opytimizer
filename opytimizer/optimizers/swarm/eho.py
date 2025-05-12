"""Elephant Herding Optimization.
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


class EHO(Optimizer):
    """An EHO class, inherited from Optimizer.

    This is the designed class to define EHO-related
    variables and methods.

    References:
        G.-G. Wang, S. Deb and L. Coelho. Elephant Herding Optimization.
        International Symposium on Computational and Business Intelligence (2015).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> EHO.")

        super(EHO, self).__init__()

        self.alpha = 0.5
        self.beta = 0.1

        self.n_clans = 10

        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Matriarch influence."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0 or alpha > 1:
            raise e.ValueError("`alpha` should be between 0 and 1")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Center influence."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0 or beta > 1:
            raise e.ValueError("`beta` should be between 0 and 1")

        self._beta = beta

    @property
    def n_clans(self) -> int:
        """Maximum number of clans."""

        return self._n_clans

    @n_clans.setter
    def n_clans(self, n_clans: int) -> None:
        if not isinstance(n_clans, int):
            raise e.TypeError("`n_clans` should be an integer")
        if n_clans < 1:
            raise e.ValueError("`n_clans` should be > 0")

        self._n_clans = n_clans

    @property
    def n_ci(self) -> int:
        """Number of elephants per clan."""

        return self._n_ci

    @n_ci.setter
    def n_ci(self, n_ci: int) -> None:
        if not isinstance(n_ci, int):
            raise e.TypeError("`n_ci` should be an integer")
        if n_ci < 1:
            raise e.ValueError("`n_ci` should be > 0")

        self._n_ci = n_ci

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.n_ci = space.n_agents // self.n_clans

    def _get_agents_from_clan(self, agents: List[Agent], index: int) -> List[Agent]:
        """Gets a set of agents from a specified clan.

        Args:
            agents: List of agents.
            index: Index of clan.

        Returns:
            (List[Agent]): A sorted list of agents that belongs to the specified clan.

        """

        start, end = index * self.n_ci, (index + 1) * self.n_ci

        if (index + 1) == self.n_clans:
            return sorted(agents[start:], key=lambda x: x.fit)

        return sorted(agents[start:end], key=lambda x: x.fit)

    def _updating_operator(
        self, agents: List[Agent], centers: np.ndarray, function: Function
    ) -> None:
        """Performs the separating operator.

        Args:
            agents: List of agents.
            centers: List of centers.
            function: A Function object that will be used as the objective function.

        """

        for i in range(self.n_clans):
            clan_agents = self._get_agents_from_clan(agents, i)
            for j, agent in enumerate(clan_agents):
                a = copy.deepcopy(agent)
                r1 = r.generate_uniform_random_number()

                if j == 0:
                    # Updates its position (eq. 2)
                    a.position = self.beta * centers[i]
                else:
                    # Updates its position (eq. 1)
                    a.position += (
                        self.alpha * (clan_agents[0].position - a.position) * r1
                    )
                a.clip_by_bound()

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def _separating_operator(self, agents: List[Agent]) -> None:
        """Performs the separating operator.

        Args:
            agents: List of agents.

        """

        for i in range(self.n_clans):
            clan_agents = self._get_agents_from_clan(agents, i)

            # Generates a new position for the worst agent in clan (eq. 4)
            worst = clan_agents[-1]
            worst.fill_with_uniform()

    def update(self, space: Space, function: Function) -> None:
        """Wraps Elephant Herd Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        centers = []

        for i in range(self.n_clans):
            clan_agents = self._get_agents_from_clan(space.agents, i)

            clan_center = np.mean(
                np.array([agent.position for agent in clan_agents]), axis=0
            )

            centers.append(clan_center)

        self._updating_operator(space.agents, centers, function)
        self._separating_operator(space.agents)
