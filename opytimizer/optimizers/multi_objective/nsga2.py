"""NSGA-II.
"""

import numpy as np

from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.utils import logging
from opytimizer.utils.operators import arithmetic_crossover, gaussian_mutation

logger = logging.get_logger(__name__)


class NSGA2(MultiObjectiveOptimizer):
    """NSGA2 class, inherited from MultiObjectiveOptimizer.

    References:
        K. Deb et al. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
        IEEE Transactions on Evolutionary Computation (2002).

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

        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA2.")

        super().__init__()

        self.crossover_rate = 0.9
        self.mutation_rate = 0.1
        self.crossover_operator = crossover_operator or arithmetic_crossover
        self.mutation_operator = mutation_operator or gaussian_mutation
        self.crossover_params = crossover_params or {}
        self.mutation_params = mutation_params or {}

        self.build(params)

        logger.info("Class overrided.")

    @property
    def rank(self) -> np.ndarray:
        """Array of ranks."""

        return self._rank

    @rank.setter
    def rank(self, rank: np.ndarray) -> None:
        if not isinstance(rank, np.ndarray):
            raise e.TypeError("`rank` should be a numpy array")

        self._rank = rank

    @property
    def crowding_distance(self) -> np.ndarray:
        """Array of crowding distances."""

        return self._crowding_distance

    @crowding_distance.setter
    def crowding_distance(self, crowding_distance: np.ndarray) -> None:
        if not isinstance(crowding_distance, np.ndarray):
            raise e.TypeError("`crowding_distance` should be a numpy array")

        self._crowding_distance = crowding_distance

    def compile(self, space: 'Space') -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Inicializa arrays com o dobro do tamanho para acomodar população combinada
        self.rank = np.zeros(space.n_agents * 2)
        self.crowding_distance = np.zeros(space.n_agents * 2)

    def _fast_non_dominated_sort(self, agents: list) -> list:
        """Performs the fast non-dominated sort.

        Args:
            agents: List of agents to be sorted.

        Returns:
            (list): List of fronts.

        """

        n_agents = len(agents)
        domination_count = np.zeros(n_agents)
        dominated_solutions = [[] for _ in range(n_agents)]
        fronts = [[]]

        # Calculates a dominance
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    if agents[i].dominates(agents[j]):
                        dominated_solutions[i].append(j)
                    elif agents[j].dominates(agents[i]):
                        domination_count[i] += 1

        # First front
        for i in range(n_agents):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # If there are no agents in the first front, put all in the first front
        if not fronts[0]:
            fronts[0] = list(range(n_agents))
            return fronts

        # Generates the other fronts
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for j in fronts[i]:
                for k in dominated_solutions[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _calculate_crowding_distance(self, front: list, agents: list) -> None:
        """Calculates the crowding distance for a front.

        Args:
            front: List of agents in the front.
            agents: List of all agents.

        """

        n_agents = len(front)
        if n_agents <= 2:
            for i in front:
                self.crowding_distance[i] = float('inf')
            return

        # Initializes the distances
        for i in front:
            self.crowding_distance[i] = 0

        # Calculates the distance for each objective
        n_objectives = len(agents[front[0]].fit)
        for obj in range(n_objectives):
            # Sorts the front by the current objective
            front.sort(key=lambda x: agents[x].fit[obj])
            
            # Defines the distances of the extremes as infinity
            self.crowding_distance[front[0]] = float('inf')
            self.crowding_distance[front[-1]] = float('inf')
            
            # Calculates the distance for the others
            f_max = agents[front[-1]].fit[obj]
            f_min = agents[front[0]].fit[obj]
            if f_max == f_min:
                continue
                
            for i in range(1, n_agents - 1):
                self.crowding_distance[front[i]] += (
                    agents[front[i + 1]].fit[obj] - agents[front[i - 1]].fit[obj]
                ) / (f_max - f_min)

    def _tournament_selection(self, agents: list, n_selections: int) -> list:
        """Performs tournament selection.

        Args:
            agents: List of agents.
            n_selections: Number of selections to be made.

        Returns:
            (list): Selected agents.

        """

        selected = []
        for _ in range(n_selections):
            # Selects two random agents
            i, j = np.random.choice(len(agents), 2, replace=False)
            
            # Compares rank and crowding distance
            if self.rank[i] < self.rank[j]:
                selected.append(i)
            elif self.rank[i] > self.rank[j]:
                selected.append(j)
            else:
                if self.crowding_distance[i] > self.crowding_distance[j]:
                    selected.append(i)
                else:
                    selected.append(j)
                    
        return selected

    def _crossover(self, parent1: 'Agent', parent2: 'Agent') -> tuple:
        """
        Performs the crossover between two parents.
        The operator used can be customized via the constructor.
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
        child1 = parent1.__class__(parent1.n_variables, parent1.n_dimensions, lb, ub)
        child2 = parent2.__class__(parent2.n_variables, parent2.n_dimensions, lb, ub)
        child1.position = c1_vec.reshape(parent1.position.shape)
        child2.position = c2_vec.reshape(parent2.position.shape)
        return child1, child2

    def _mutation(self, agent: 'Agent') -> None:
        """
        Performs the mutation on an agent.
        The operator used can be customized via the constructor.
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
                x, lb, ub, self.mutation_rate, **self.mutation_params
            )
        agent.position = mutant.reshape(agent.position.shape)
        agent.clip_by_bound()

    def update(self, space: 'Space') -> None:
        """Wraps NSGA-II over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Generates the offspring population (Q)
        offspring = []
        while len(offspring) < len(space.agents):
            # Selects parents
            parents = self._tournament_selection(space.agents, 2)
            parent1, parent2 = space.agents[parents[0]], space.agents[parents[1]]
            
            # Performs crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Performs mutation
            self._mutation(child1)
            self._mutation(child2)
            
            offspring.extend([child1, child2])
        
        # Combines current population (P) and offspring (Q)
        combined_population = space.agents + offspring[:len(space.agents)]
        
        # Performs the fast non-dominated sort
        fronts = self._fast_non_dominated_sort(combined_population)
        
        # Assigns ranks and calculates crowding distance
        for i, front in enumerate(fronts):
            for j in front:
                self.rank[j] = i
            self._calculate_crowding_distance(front, combined_population)
        
        # Selects the new population
        selected = self._tournament_selection(combined_population, len(space.agents))
        space.agents = [combined_population[i] for i in selected]