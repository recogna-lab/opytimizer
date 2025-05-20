"""NSGA-II.
"""

import numpy as np
import copy

import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
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
        self.mutation_rate = 0.025
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
        
        self.rank = np.zeros(space.n_agents)
        self.crowding_distance = np.zeros(space.n_agents)

    def _fast_non_dominated_sort(self, agents: list) -> list:
        """Performs the fast non-dominated sort.

        Args:
            agents: List of agents to be sorted.

        Returns:
            (list): List of fronts.

        """

        n_agents = len(agents)
        domination_count = np.zeros(n_agents,dtype=int)
        dominated_solutions = [[] for _ in range(n_agents)]
        fronts = [[]]

        # Calculates a dominance
        for i in range(n_agents):
            for j in range(i+1,n_agents):
                
                if agents[i].dominates(agents[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif agents[j].dominates(agents[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
                    
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Generates the other fronts
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for j in fronts[i]:
                for k in dominated_solutions[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            if next_front:
                fronts.append(next_front)
            i += 1
                
        #Assign ranks
        self.rank=np.zeros(n_agents,dtype=int)
        for rank_, front in enumerate(fronts):
            for i in front:
                self.rank[i] = rank_
                
        return fronts

    def _calculate_crowding_distance(self, front: list, agents: list) -> np.ndarray:
        """Calculates the crowding distance for a front.

        Args:
            front: List of agents in the front.
            agents: List of all agents.
            
        Returns:
            (np.ndarray): Crowding distance for the front.

        """
        
        distances = np.zeros(len(front))
        if len(front) == 0:
            return distances
       
        fitness_mtx = np.array([agents[i].fit for i in front])
        
        n_objectives = fitness_mtx.shape[1]
        
        for m in range(n_objectives):
            sorted_indices = np.argsort(fitness_mtx[:, m])
            # Sorts the front based on the current objective
            sorted_front = np.array(front)[sorted_indices]
            
            #Boundary points
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]]= float('inf')
            # Normalizes the distance
            min_val = fitness_mtx[sorted_indices[0],m]
            max_val = fitness_mtx[sorted_indices[-1],m]
            norm= (max_val - min_val) if max_val > min_val else 1.
            
            for i in range(1, len(front) - 1):
                prev_idx = int(sorted_front[i - 1])
                next_idx = int(sorted_front[i + 1])
                prev_fit = agents[prev_idx].fit[m]
                next_fit = agents[next_idx].fit[m]
                distances[sorted_indices[i]] += (next_fit - prev_fit) / norm
                
        return distances

    def _tournament_selection(self, agents: list) -> list:
        """Performs tournament selection.

        Args:
            agents: List of agents.
            n_selections: Number of selections to be made.

        Returns:
            (list): Selected agents.

        """

        selected = []
        
        for _ in range(len(agents)):
            # Selects two random agents
            i, j = np.random.choice(len(agents), 2, replace=False)
            
            # Compares rank and crowding distance
            if self.rank[i] < self.rank[j]:
                winner=i
            elif self.rank[i] > self.rank[j]:
                winner=j
            else:
                if self.crowding_distance[i] > self.crowding_distance[j]:
                    winner=i
                else:
                    winner=j
            selected.append(agents[winner])
        
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
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.position = c1_vec.reshape(parent1.position.shape)
        child2.position = c2_vec.reshape(parent2.position.shape)
        
        return child1, child2

    def _mutation(self, agent: 'Agent') -> Agent:
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
            
        mutated = copy.deepcopy(agent)
        mutated.position = mutant.reshape(agent.position.shape)
        mutated.clip_by_bound()
        
        return mutated

    def _create_offspring(self, space: 'Space') -> list[np.ndarray]:
         """Generates offspring using SBX crossover and PM mutation."""
         parents=self._tournament_selection(space.agents)
         offspring=[]
         for i in range(0, len(parents), 2):
                parent1=parents[i]
                parent2=parents[i+1] if i+1<len(space.agents) else parents[0]
                child1, child2=self._crossover(parent1,parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                offspring.extend([child1, child2])
         return offspring[:len(space.agents)]
     
    def _select_survivors(self, combined_population: list,space,function) -> list:
        """Selects the next generation of agents based on non-dominated sorting and crowding distance.

        Args:
            combined_population: Combined population of parents and offspring.
            space: Space containing agents and update-related information.
            function: Function to evaluate the fitness of the agents.

        Returns:
            (list): Selected agents for the next generation.

        """
        fronts = self._fast_non_dominated_sort(combined_population)
       
        new_population = []
        n_agents = len(space.agents)

        # Assigns ranks and calculates crowding distance
        for front in fronts:
            if len(new_population) >= n_agents: break
            
            crowding=self._calculate_crowding_distance(front, combined_population)
            sorted_front = sorted(zip(front, crowding),key=lambda x:x[1],reverse=True)
            sorted_indices=[idx for idx,_ in sorted_front]
            
            remaining = n_agents - len(new_population)
            new_population.extend([combined_population[i] for i in sorted_indices[:remaining]])

        new_population=new_population[:n_agents]
        
        return new_population
    
    def update(self, space: 'Space',function) -> None:
        """Wraps NSGA-II over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """
      
        offspring=self._create_offspring(space)
        for i in range(len(offspring)):
            offspring[i].fit = function(offspring[i].position)
            
        combined_population = space.agents + offspring
       
        # Generates the offspring population (Q)
        new_pop=self._select_survivors(combined_population,space,function)

        # Updates the population with the new agents
        for i in range(len(space.agents)):
            space.agents[i] = new_pop[i]
            
        
    def evaluate(self, space, function):
        for agent in space.agents:
            agent.fit = function(agent.position)
        
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(space.agents)
        self.crowding_distance = np.zeros(len(space.agents))
        for front in fronts:
            if not front: continue
            front_indices = np.array(front)
            self.crowding_distance[front_indices] = self._calculate_crowding_distance(front, space.agents)
        
        # Updates the Pareto front
        self.update_pareto_front(space.agents)
       