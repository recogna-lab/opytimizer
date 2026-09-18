"""Genetic Programming (Strongly-Typed).
"""

import copy
import time
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.graph.operators import crossover_trees, mutate_tree
from opytimizer.core.graph.space import _SingleObjectiveSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GP(Optimizer):
    """A GP class, inherited from Optimizer.

    Adapted to support Strongly-Typed Genetic Programming using Tree/Graph models.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """
        logger.info("Overriding class: Optimizer -> GP.")

        super(GP, self).__init__()

        self.p_reproduction = 0.25
        self.p_mutation = 0.1
        self.p_crossover = 0.1

        self.build(params)

        logger.info("Class overrided.")

    @property
    def p_reproduction(self) -> float:
        """Probability of reproduction."""
        return self._p_reproduction

    @p_reproduction.setter
    def p_reproduction(self, p_reproduction: float) -> None:
        if not isinstance(p_reproduction, (float, int)):
            raise e.TypeError("`p_reproduction` should be a float or integer")
        if p_reproduction < 0 or p_reproduction > 1:
            raise e.ValueError("`p_reproduction` should be between 0 and 1")
        self._p_reproduction = p_reproduction

    @property
    def p_mutation(self) -> float:
        """Probability of mutation."""
        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if not isinstance(p_mutation, (float, int)):
            raise e.TypeError("`p_mutation` should be a float or integer")
        if p_mutation < 0 or p_mutation > 1:
            raise e.ValueError("`p_mutation` should be between 0 and 1")
        self._p_mutation = p_mutation

    @property
    def p_crossover(self) -> float:
        """Probability of crossover."""
        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, p_crossover: float) -> None:
        if not isinstance(p_crossover, (float, int)):
            raise e.TypeError("`p_crossover` should be a float or integer")
        if p_crossover < 0 or p_crossover > 1:
            raise e.ValueError("`p_crossover` should be between 0 and 1")
        self._p_crossover = p_crossover

    def _reproduction(self, space: _SingleObjectiveSpace) -> None:
        """Reproduces a number of individuals pre-selected through a tournament procedure.

        Args:
            space: A TreeSpace object.

        """
        fitness = [agent.fit for agent in space.agents]
        n_individuals = int(space.n_agents * self.p_reproduction)

        selected = g.tournament_selection(fitness, n_individuals)
        for s in selected:
            worst = int(np.argmax(fitness))

            # The agent's position is exactly the Tree instance
            space.agents[worst].position = space.agents[s].position.copy()
            fitness[worst] = 0

    def _mutation(self, space: _SingleObjectiveSpace) -> None:
        """Mutates a number of individuals pre-selected through a tournament procedure.

        Args:
            space: A TreeSpace object.

        """
        fitness = [agent.fit for agent in space.agents]
        n_individuals = int(space.n_agents * self.p_mutation)

        selected = g.tournament_selection(fitness, n_individuals)
        for s in selected:
            agent = space.agents[s]

            # Safely mutates the tree respecting typings
            mutate_tree(agent.position, space.pset, max_depth=space.max_depth)

    def _crossover(self, space: _SingleObjectiveSpace) -> None:
        """Crossover a number of individuals pre-selected through a tournament procedure.

        Args:
            space: A TreeSpace object.

        """
        fitness = [agent.fit for agent in space.agents]

        n_individuals = int(space.n_agents * self.p_crossover)
        if n_individuals % 2 != 0:
            n_individuals += 1

        selected = g.tournament_selection(fitness, n_individuals)
        for s in g.n_wise(selected):
            if s[0] == s[1]:
                continue
            agent_father = space.agents[s[0]]
            agent_mother = space.agents[s[1]]

            # In-place crossover exchanging compatible typed branches
            crossover_trees(agent_father.position, agent_mother.position)

    def evaluate(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A TreeSpace object.
            function: A Function object that will be used as the objective function.

        """
        for agent in space.agents:

            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = agent.position.copy()
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: _SingleObjectiveSpace) -> None:
        """Wraps Genetic Programming over all trees and variables.

        Args:
            space: TreeSpace containing agents and update-related information.

        """
        self._reproduction(space)
        self._crossover(space)
        self._mutation(space)
