import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation

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
    ) -> None:
        """Initialization method."""

        logger.info("Overriding class: MultiObjectiveOptimizer -> SPEA2.")

        super().__init__()

        self.archive_size = 100
        self.crossover_operator = crossover_operator or SBXCrossover(return_mode='both')
        self.mutation_operator = mutation_operator or PolynomialMutation()

        self.build(params)

        logger.info("Class overrided.")

    @property
    def archive_size(self) -> int:
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
        return self._strength

    @strength.setter
    def strength(self, strength: np.ndarray) -> None:
        if not isinstance(strength, np.ndarray):
            raise e.TypeError("`strength` should be a numpy array")
        self._strength = strength

    @property
    def raw_fitness(self) -> np.ndarray:
        return self._raw_fitness

    @raw_fitness.setter
    def raw_fitness(self, raw_fitness: np.ndarray) -> None:
        if not isinstance(raw_fitness, np.ndarray):
            raise e.TypeError("`raw_fitness` should be a numpy array")
        self._raw_fitness = raw_fitness

    @property
    def density(self) -> np.ndarray:
        return self._density

    @density.setter
    def density(self, density: np.ndarray) -> None:
        if not isinstance(density, np.ndarray):
            raise e.TypeError("`density` should be a numpy array")
        self._density = density

    def compile(self, space: "Space") -> None:
        """Compiles additional information that is used by this optimizer."""
        self.strength = np.zeros(space.n_agents)
        self.raw_fitness = np.zeros(space.n_agents)
        self.density = np.zeros(space.n_agents)

    def _calculate_strength(self, agents: list) -> None:
        """Calculates the strength of each solution (S(i))."""
        n_agents = len(agents)
        self.strength = np.zeros(n_agents)

        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and agents[i].dominates(agents[j]):
                    self.strength[i] += 1

    def _calculate_raw_fitness(self, agents: list) -> None:
        """Calculates the raw fitness of each solution (R(i))."""
        n_agents = len(agents)
        self.raw_fitness = np.zeros(n_agents)

        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and agents[j].dominates(agents[i]):
                    self.raw_fitness[i] += self.strength[j]

    def _calculate_density(self, agents: list) -> None:
        """Calculates the density of each solution using k-nearest neighbor."""
        n_agents = len(agents)
        # k = sqrt(N + N') roughly, usually sqrt(N) is used as approx
        k = int(np.sqrt(n_agents))
        self.density = np.zeros(n_agents)

        distances = np.zeros((n_agents, n_agents))
        
        # Pre-compute fitness vectors for speed
        fits = np.array([a.fit for a in agents])
        
   
        for i in range(n_agents):
            
            d = np.linalg.norm(fits[i] - fits, axis=1)
            distances[i] = d

        for i in range(n_agents):
            # Sort distances (ascending)
            sorted_dists = np.sort(distances[i])
            
            # The 0-th distance is to itself (0.0), so we look at index k
            if k < len(sorted_dists):
                k_dist = sorted_dists[k]
            else:
                k_dist = sorted_dists[-1]
            
            self.density[i] = 1.0 / (k_dist + 2.0)

    def _update_metrics(self, agents: list) -> None:
        """Helper to update all SPEA2 metrics for a given list of agents."""
        self._calculate_strength(agents)
        self._calculate_raw_fitness(agents)
        self._calculate_density(agents)

    def _environmental_selection(self, agents: list) -> list:
        """Performs environmental selection to maintain the archive."""
        
        # Calculate metrics for the combined population (P + Q)
        self._update_metrics(agents)

        fitness = self.raw_fitness + self.density

      
        non_dominated_indices = [i for i, f in enumerate(self.raw_fitness) if f == 0.0]

        if len(non_dominated_indices) > self.archive_size:
            
            sorted_indices = sorted(non_dominated_indices, key=lambda idx: self.density[idx])
            
            selected_indices = sorted_indices[:self.archive_size]
            
        elif len(non_dominated_indices) < self.archive_size:
            # Fill with dominated solutions
            selected_indices = list(non_dominated_indices)
            remaining_slots = self.archive_size - len(non_dominated_indices)
            
            dominated_indices = [i for i in range(len(agents)) if i not in non_dominated_indices]
            
            # Sort dominated by total fitness (lower is better)
            sorted_dominated = sorted(dominated_indices, key=lambda idx: fitness[idx])
            
            selected_indices.extend(sorted_dominated[:remaining_slots])
        else:
            selected_indices = non_dominated_indices

        return [agents[i] for i in selected_indices]

    def _tournament_selection(self, agents: list) -> list:
        """Binary tournament selection based on SPEA2 fitness."""
        selected = []
        n_agents = len(agents)
        
        # Total fitness = Raw + Density
        # Assumes self.raw_fitness/density correspond to 'agents' list order
        fitness = self.raw_fitness + self.density

        for _ in range(n_agents):
            i, j = np.random.choice(n_agents, 2, replace=False)
            
            if fitness[i] < fitness[j]:
                winner = i
            else:
                winner = j
                
            selected.append(agents[winner])

        return selected

    def _crossover(self, parent1: "Agent", parent2: "Agent") -> tuple:
        return self.crossover_operator(parent1, parent2)

    def _mutation(self, agent: "Agent") -> Agent:
        return self.mutation_operator(agent)

    def _create_offspring(self, space: "Space") -> list:
        parents = self._tournament_selection(space.agents)
        offspring = []

        for i in range(0, len(parents), 2):
            # wrap around if odd number of parents
            parent1 = parents[i]
            parent2 = parents[i + 1] if (i + 1) < len(parents) else parents[0]
            
            children = self._crossover(parent1, parent2)
            for child in children:
                offspring.append(self._mutation(child))

        return offspring[: len(space.agents)]

    def update(self, space: "Space", function) -> None:
        """Wraps SPEA2 over all agents."""
        
        
        self._update_metrics(space.agents)

      
        offspring = self._create_offspring(space)
        
       
        for agent in offspring:
            agent.fit = function(agent.position)

      
        combined_population = space.agents + offspring
        new_pop = self._environmental_selection(combined_population)

      
        space.agents = new_pop
        
      