"""Genetic Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.general as g
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import _SingleObjectiveSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import ArithmeticCrossover, GaussianMutation

logger = logging.get_logger(__name__)


class GA(Optimizer):
    """An GA class, inherited from Optimizer.

    This is the designed class to define GA-related
    variables and methods.

    References:
        M. Mitchell. An introduction to genetic algorithms. MIT Press (1998).

    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        crossover_operator=None,
        mutation_operator=None,
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.
        """

        super(GA, self).__init__()

        self.p_selection = 0.75
        self.crossover_operator = crossover_operator or ArithmeticCrossover()
        self.mutation_operator = mutation_operator or GaussianMutation()

        self.build(params)

        logger.info("Class overrided.")

    @property
    def p_selection(self) -> float:
        """Probability of selection."""

        return self._p_selection

    @p_selection.setter
    def p_selection(self, p_selection: float) -> None:
        if not isinstance(p_selection, (float, int)):
            raise e.TypeError("`p_selection` should be a float or integer")
        if p_selection < 0.0 or p_selection > 1.0:
            raise e.ValueError("`p_selection` should be between 0 and 1")

        self._p_selection = p_selection


    def _roulette_selection(self, n_agents: int, fitness: List[float]) -> List[int]:
        """Performs a roulette selection on the population (p. 8).

        Args:
            n_agents: Number of agents allowed in the space.
            fitness: A fitness list of every agent.

        Returns:
            (List[int]): The selected indexes of the population.

        """

        n_individuals = int(n_agents * self.p_selection)
        if n_individuals % 2 != 0:
            n_individuals += 1

        max_fitness = np.max(fitness)

        # Re-arrange the list of fitness by inverting it
        # Note that we apply a trick due to it being designed for minimization
        # f'(x) = f_max - f(x)
        inv_fitness = max_fitness - fitness + c.EPSILON
        total_fitness = np.sum(inv_fitness)

        probs = [fit / total_fitness for fit in inv_fitness]

        selected = d.generate_choice_distribution(n_agents, probs, n_individuals)

        return selected

    def _crossover(self, father: Agent, mother: Agent) -> Tuple[Agent, Agent]:
        """Performs the crossover between a pair of parents (p. 8).

        Args:
            father: Father to produce the offsprings.
            mother: Mother to produce the offsprings.

        Returns:
            (Tuple[Agent, Agent]): Two generated offsprings based on parents.

        """

        
        child1, child2 = self.crossover_operator(father, mother)
        
        return child1[0], child2[0]

    def _mutation(self, agent: Agent) -> Agent:
        """Performs the mutation over offsprings (p. 8).

        Args:
            alpha: First offspring.
            beta: Second offspring.

        Returns:
            (Tuple[Agent, Agent]): Two mutated offsprings.

        """

        
        mutated = self.mutation_operator(agent)
       

        mutated.clip_by_bound()
        return mutated

    def update(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """Wraps Genetic Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        new_agents = []
        n_agents = len(space.agents)

        fitness = [agent.fit + c.EPSILON for agent in space.agents]

        selected = self._roulette_selection(n_agents, fitness)
        for s in g.n_wise(selected):
            parent1 = space.agents[s[0]]
            parent2 = space.agents[s[1]]
            children = self._crossover(parent1, parent2)
            child1, child2 = children
           
            child1 = self._mutation(child1)
            
            child2 = self._mutation(child2)

            child1.fit = function(child1.position)
            child2.fit = function(child2.position)


            new_agents.extend([child1, child2])

        space.agents += new_agents
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]
