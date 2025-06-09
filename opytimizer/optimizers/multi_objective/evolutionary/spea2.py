"""SPEA2."""

import copy

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging
from opytimizer.utils.operators import arithmetic_crossover, gaussian_mutation

logger = logging.get_logger(__name__)


class SPEA2(MultiObjectiveOptimizer):
    """SPEA2 class, inherited from MultiObjectiveOptimizer.

    References:
        E. Zitzler et al. SPEA2: Improving the Strength Pareto Evolutionary Algorithm.
        Technical Report 103, Computer Engineering and Networks Laboratory (2001).

    """

    def __init__(
        self,
        params: dict = None,
        crossover_operator=None,
        mutation_operator=None,
        crossover_params=None,
        mutation_params=None,
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.
            crossover_params: Parameters for the crossover operator.
            mutation_params: Parameters for the mutation operator.

        """

        logger.info("Overriding class: MultiObjectiveOptimizer -> SPEA2.")

        super().__init__()

        self.crossover_rate = 0.9
        self.mutation_rate = 0.025
        self.archive_size = 100
        self.crossover_operator = crossover_operator or arithmetic_crossover
        self.mutation_operator = mutation_operator or gaussian_mutation
        self.crossover_params = crossover_params or {}
        self.mutation_params = mutation_params or {}

        self.build(params)

        logger.info("Class overrided.")

    @property
    def crossover_rate(self) -> float:
        """Probability of crossover."""

        return self._crossover_rate

    @crossover_rate.setter
    def crossover_rate(self, crossover_rate: float) -> None:
        if not isinstance(crossover_rate, (float, int)):
            raise e.TypeError("`crossover_rate` should be a float or integer")
        if crossover_rate < 0 or crossover_rate > 1:
            raise e.ValueError("`crossover_rate` should be between 0 and 1")

        self._crossover_rate = crossover_rate

    @property
    def mutation_rate(self) -> float:
        """Probability of mutation."""

        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, mutation_rate: float) -> None:
        if not isinstance(mutation_rate, (float, int)):
            raise e.TypeError("`mutation_rate` should be a float or integer")
        if mutation_rate < 0 or mutation_rate > 1:
            raise e.ValueError("`mutation_rate` should be between 0 and 1")

        self._mutation_rate = mutation_rate

    @property
    def archive_size(self) -> int:
        """Size of the external archive."""

        return self._archive_size

    @archive_size.setter
    def archive_size(self, archive_size: int) -> None:
        if not isinstance(archive_size, int):
            raise e.TypeError("`archive_size` should be an integer")
        if archive_size < 0:
            raise e.ValueError("`archive_size` should be >= 0")

        self._archive_size = archive_size

    @property
    def strength(self) -> np.ndarray:
        """Array of strength values."""

        return self._strength

    @strength.setter
    def strength(self, strength: np.ndarray) -> None:
        if not isinstance(strength, np.ndarray):
            raise e.TypeError("`strength` should be a numpy array")

        self._strength = strength

    @property
    def raw_fitness(self) -> np.ndarray:
        """Array of raw fitness values."""

        return self._raw_fitness

    @raw_fitness.setter
    def raw_fitness(self, raw_fitness: np.ndarray) -> None:
        if not isinstance(raw_fitness, np.ndarray):
            raise e.TypeError("`raw_fitness` should be a numpy array")

        self._raw_fitness = raw_fitness

    @property
    def density(self) -> np.ndarray:
        """Array of density values."""

        return self._density

    @density.setter
    def density(self, density: np.ndarray) -> None:
        if not isinstance(density, np.ndarray):
            raise e.TypeError("`density` should be a numpy array")

        self._density = density

    def compile(self, space: "Space") -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.strength = np.zeros(space.n_agents)
        self.raw_fitness = np.zeros(space.n_agents)
        self.density = np.zeros(space.n_agents)

    def _calculate_strength(self, agents: list) -> None:
        """Calculates the strength of each solution.

        Args:
            agents: List of agents.

        """

        n_agents = len(agents)
        self.strength = np.zeros(n_agents)

        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and agents[i].dominates(agents[j]):
                    self.strength[i] += 1

    def _calculate_raw_fitness(self, agents: list) -> None:
        """Calculates the raw fitness of each solution.

        Args:
            agents: List of agents.

        """

        n_agents = len(agents)
        self.raw_fitness = np.zeros(n_agents)

        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and agents[j].dominates(agents[i]):
                    self.raw_fitness[i] += self.strength[j]

    def _calculate_density(self, agents: list) -> None:
        """Calculates the density of each solution using k-nearest neighbor.

        Args:
            agents: List of agents.

        """

        n_agents = len(agents)
        k = int(np.sqrt(n_agents))
        self.density = np.zeros(n_agents)

        # Creates a matrix of distances between all solutions
        distances = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = np.linalg.norm(agents[i].fit - agents[j].fit)
                distances[i, j] = dist
                distances[j, i] = dist

        # For each solution, calculates its density
        for i in range(n_agents):
            # Gets the k-th nearest neighbor distance
            k_dist = np.sort(distances[i])[k]
            self.density[i] = 1 / (k_dist + 2)

    def _environmental_selection(self, agents: list) -> list:
        """Performs environmental selection to maintain the archive.

        Args:
            agents: List of agents.

        Returns:
            (list): Selected agents for the archive.

        """

        # Calculates strength, raw fitness and density
        self._calculate_strength(agents)
        self._calculate_raw_fitness(agents)
        self._calculate_density(agents)

        # Calculates the final fitness
        fitness = self.raw_fitness + self.density

        # Selects non-dominated solutions
        non_dominated = []
        for i in range(len(agents)):
            if self.raw_fitness[i] < 1:
                non_dominated.append(i)

        # If we have more non-dominated solutions than the archive size
        if len(non_dominated) > self.archive_size:
            # Sorts by density
            sorted_indices = np.argsort(self.density[non_dominated])
            selected = [non_dominated[i] for i in sorted_indices[: self.archive_size]]
        else:
            # If we have less non-dominated solutions than the archive size
            selected = non_dominated
            remaining = self.archive_size - len(non_dominated)

            # Gets the remaining solutions from the dominated ones
            dominated = [i for i in range(len(agents)) if i not in non_dominated]
            sorted_indices = np.argsort(fitness[dominated])
            selected.extend([dominated[i] for i in sorted_indices[:remaining]])

        return [agents[i] for i in selected]

    def _tournament_selection(self, agents: list) -> list:
        """Performs tournament selection.

        Args:
            agents: List of agents.

        Returns:
            (list): Selected agents.

        """

        selected = []

        for _ in range(len(agents)):
            # Selects two random agents
            i, j = np.random.choice(len(agents), 2, replace=False)

            # Compares fitness
            if self.raw_fitness[i] < self.raw_fitness[j]:
                winner = i
            elif self.raw_fitness[i] > self.raw_fitness[j]:
                winner = j
            else:
                if self.density[i] < self.density[j]:
                    winner = i
                else:
                    winner = j
            selected.append(agents[winner])

        return selected

    def _crossover(self, parent1: "Agent", parent2: "Agent") -> tuple:
        """Performs the crossover between two parents.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            (tuple): Two children.

        """

        # Gets the vectors of the parents
        p1 = parent1.position.flatten()
        p2 = parent2.position.flatten()

        lb = parent1.lb
        ub = parent1.ub

        # Applies the custom operator
        if self.crossover_operator == arithmetic_crossover:
            c1_vec, c2_vec = self.crossover_operator(
                p1, p2, self.crossover_rate, **self.crossover_params
            )
        else:
            c1_vec, c2_vec = self.crossover_operator(
                p1, p2, lb, ub, self.crossover_rate, **self.crossover_params
            )

        # Creates new agents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.position = c1_vec.reshape(parent1.position.shape)
        child2.position = c2_vec.reshape(parent2.position.shape)

        return child1, child2

    def _mutation(self, agent: "Agent") -> Agent:
        """Performs the mutation on an agent.

        Args:
            agent: Agent to be mutated.

        Returns:
            (Agent): Mutated agent.

        """

        x = agent.position.flatten()
        lb = agent.lb
        ub = agent.ub

        if self.mutation_operator == gaussian_mutation:
            mutant = self.mutation_operator(
                x, self.mutation_rate, **self.mutation_params
            )
        else:
            mutant = self.mutation_operator(
                vector=x,
                lb=lb,
                ub=ub,
                mutation_rate=self.mutation_rate,
                **self.mutation_params
            )

        mutated = copy.deepcopy(agent)
        mutated.position = mutant.reshape(agent.position.shape)
        mutated.clip_by_bound()

        return mutated

    def _create_offspring(self, space: "Space") -> list:
        """Generates offspring using crossover and mutation.

        Args:
            space: Space containing agents and offspring-related information.

        Returns:
            (list): Offspring agents.

        """

        parents = self._tournament_selection(space.agents)
        offspring = []

        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(space.agents) else parents[0]
            child1, child2 = self._crossover(parent1, parent2)
            child1 = self._mutation(child1)
            child2 = self._mutation(child2)
            offspring.extend([child1, child2])

        return offspring[: len(space.agents)]

    def update(self, space: "Space", function) -> None:
        """Wraps SPEA2 over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: Function to evaluate the fitness of the agents.

        """

        # Creates offspring population
        offspring = self._create_offspring(space)
        for i in range(len(offspring)):
            offspring[i].fit = function(offspring[i].position)

        # Combines parent and offspring populations
        combined_population = space.agents + offspring

        # Performs environmental selection
        new_pop = self._environmental_selection(combined_population)

        # Updates the population
        space.agents = new_pop
