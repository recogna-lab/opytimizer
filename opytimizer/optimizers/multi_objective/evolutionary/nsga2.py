"""NSGA-II."""

from __future__ import annotations

 
import numpy as np

from typing import List, Tuple, Any, Dict
import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer, Environment, TensorizedMultiObjectiveOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import _MultiObjectiveSpace, _MultiObjectiveTensorSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation, SBXCrossoverTensor, PolynomialMutationTensor
from opytimizer.math.random import generate_integer_random_number 

logger = logging.get_logger(__name__)


class NSGA2(MultiObjectiveOptimizer):
    """NSGA2 class, inherited from MultiObjectiveOptimizer.

    References:
        K. Deb et al. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
        IEEE Transactions on Evolutionary Computation (2002).

    """

    def __init__(
        self,
        params: Dict = None,
        crossover_operator=None,
        mutation_operator=None,
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.

        """

        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA2 (Default).")

        super().__init__()

    
        self.crossover_operator = crossover_operator or SBXCrossover()
        self.mutation_operator = mutation_operator or PolynomialMutation()

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

    def compile(self, space: _MultiObjectiveSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        
        self.crowding_distance = np.zeros(space.n_agents)

    def _fast_non_dominated_sort(self, agents:List[Agent]) -> List:
        """Performs the fast non-dominated sort.

        Args:
            agents: List of agents to be sorted.

        Returns:
            (List): List of fronts.

        """

        n = len(agents)
        # fit extraction for numpy broadcasting -> matricial operations
        population_fitness = [ag.fit for ag in agents] # (N, M) where N=number of agents; and M=number of objectives
        population_fitness = np.array(population_fitness)
        P = population_fitness[:, np.newaxis, :] # (N, 1, M)
        Q = population_fitness[np.newaxis, :, :] # (1, N, M)

        # pareto dominance operation
        # a solution X dominates Y if, and only if, all objectives m_{x} <= m_{y} and
        # m_{x} < m_{y} on at least one objective 

        cond_1 = np.all(P <= Q, axis=2) # (N, N)
        cond_2 = np.any(P < Q, axis=2) # (N, N)
        
        dominance_matrix = cond_1 & cond_2 # (N, N)

        # A_{ij} = True means that agent i dominates agent j

        num_dominance = np.sum(dominance_matrix, axis=0) # (N, )
        # np.sum() == 0 means that no other solution dominates agent j

        local_rank = np.zeros(n, dtype=int)

        current_front = np.where(num_dominance == 0)[0].tolist()
        fronts = [current_front]

        local_rank[current_front] = 0

        # crucial for tournament selection
        aux_rank = 1

        while len(current_front) > 0:
            next_front = []
            for p in current_front:
                dominated_by_p = np.where(dominance_matrix[p])[0]

                for q in dominated_by_p:
                    num_dominance[q] -= 1
                    if num_dominance[q] == 0:
                        next_front.append(q)
                        local_rank[q] = aux_rank
            
            if len(next_front) > 0:
                fronts.append(next_front)
            
            current_front = next_front
            aux_rank += 1

        return fronts, local_rank # return a list of agents indices in each front
    
    def _calculate_crowding_distance(self, front: List, agents: List[Agent]) -> np.ndarray:
        """Calculates the crowding distance for a front.

        Args:
            front: List of agents indices in the front.
            agents: List of all agents.

        Returns:
            (np.ndarray): Crowding distance for the front.

        """
        
        n_agents_in_front = len(front)
        distances = np.zeros(n_agents_in_front)
        n_objectives = len(agents[0].fit)
        fit_values = np.array([ag.fit for ag in agents])

        if len(front) <= 2:
            distances[:]= np.inf
            return distances

        for m in range(n_objectives):
            objective_values = fit_values[front, m]

            sorted_indices = np.argsort(objective_values)

            # edge agents -> crowding distance = inf
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            amplitude = objective_values[sorted_indices[-1]] - objective_values[sorted_indices[0]]

            # avoid division by zero

            if amplitude == 0:
                 continue
            
            distances[sorted_indices[1:-1]] += (objective_values[sorted_indices[2:]] - objective_values[sorted_indices[:-2]]) / amplitude

        return distances

    def _tournament_selection(self, agents: List[Agent]) -> np.ndarray:
        """Performs tournament selection.

        Args:
            agents: List of agents.
            n_selections: Number of selections to be made.

        Returns:
            (np.ndarray): List of selected parents indices.

        """

        N = len(agents)

        competitors_A = generate_integer_random_number(low=0, high=N, size=N)
        competitors_B = generate_integer_random_number(low=0, high=N, size=N)

        # Condition 1: A has better ranking than B
        cond_ranking = self.rank[competitors_A] < self.rank[competitors_B]

        # Condition 2: Ranking tie. A has higher crowd distance than B
        cond_tie = (self.rank[competitors_A] == self.rank[competitors_B]) & (self.crowding_distance[competitors_A] > self.crowding_distance[competitors_B]) 

        victory_A = cond_ranking | cond_tie
        
        selected_parents = np.where(victory_A, competitors_A, competitors_B)

        return selected_parents

        

    def _crossover(self, parent1: Agent, parent2: Agent) -> Tuple:
        """Performs the crossover between two parents.

        The operator used can be customized via the constructor.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            (tuple): Two children.
        """
        
        children = self.crossover_operator(parent1, parent2)
        
        return children

    def _mutation(self, agent: Agent) -> Agent:
        """Performs the mutation on an agent.

        The operator used can be customized via the constructor.

        Args:
            agent: Agent to be mutated.

        Returns:
            (Agent): Mutated agent.
        """
        mutated = self.mutation_operator(agent)
        return mutated

    def _create_offspring(self, space: _MultiObjectiveSpace) -> List[Agent]:
        """Generates offspring using SBX crossover and PM mutation.

        Args:
            space: Space containing agents and offspring-related information.

        Returns:
            (list): Offspring agents.
        """

        parents_idx = self._tournament_selection(space.agents)

        n = len(space.agents)
        half = n // 2
        
        parents1 = [space.agents[int(i)] for i in parents_idx[:half]]
        parents2 = [space.agents[int(i)] for i in parents_idx[half : half * 2]]
        
        # batched crossover
        crossed = self.crossover_operator(parents1, parents2)

        # batched mutation
        mutated = self.mutation_operator(crossed)

        return mutated
    def _select_survivors(self, combined_population: List[Agent], space: _MultiObjectiveSpace):
        fronts, _ = self._fast_non_dominated_sort(combined_population)

        new_population = []
        new_ranks = []
        new_crowding = []
        n_agents = len(space.agents)

        for front_idx, front in enumerate(fronts):
            if len(new_population) >= n_agents:
                break

            crowding = self._calculate_crowding_distance(front, combined_population)
            sorted_front = sorted(
                zip(front, crowding), key=lambda x: x[1], reverse=True
            )

            remaining = n_agents - len(new_population)
            for idx, cd in sorted_front[:remaining]:
                new_population.append(combined_population[idx])
                new_ranks.append(front_idx)
                new_crowding.append(cd)

        return new_population, np.array(new_ranks), np.array(new_crowding)
    

    def update(self, space: _MultiObjectiveSpace, function) -> None:
        offspring = self._create_offspring(space)

        agent_shape = space.agents[0].position.shape
        n_offspring = min(len(offspring), len(space.agents))

        for i in range(n_offspring):
            pos = offspring[i].position.reshape(agent_shape)
            
            offspring[i].position = pos
            offspring[i].fit = function(offspring[i].position).reshape(-1)

        combined_population = space.agents + list(offspring)[:n_offspring]

        
        new_pop, new_ranks, new_crowding = self._select_survivors(combined_population, space)

        for i in range(len(space.agents)):
            space.agents[i] = new_pop[i]
            

        self.rank = new_ranks
        self.crowding_distance = new_crowding

    def evaluate(self, space, function):
        """Evaluates the fitness of the agents.

        Args:
            space: Space containing agents and evaluation-related information.
            function: Function to evaluate the fitness of the agents.

        """
        
        for agent in space.agents:
            agent.fit = function(agent.position).squeeze()
       
        # Non-dominated sorting
        fronts, local_rank = self._fast_non_dominated_sort(space.agents)
        self.rank = local_rank
        self.crowding_distance = np.zeros(len(space.agents))
        
        for front in fronts:
            if not front:
                continue
            front_indices = np.array(front)
            self.crowding_distance[front_indices] = self._calculate_crowding_distance(
                front, space.agents
            )

        self.evaluate = lambda : None


class NSGA2Tensor(MultiObjectiveOptimizer, TensorizedMultiObjectiveOptimizer):
    """Tensorized NSGA-II, following the general tensorization methodology of:

        Z. Liang, H. Li, N. Yu, K. Sun, and R. Cheng, "Bridging Evolutionary
        Multiobjective Optimization and GPU Acceleration via Tensorization,"
        IEEE Trans. Evol. Comput., vol. 30, no. 1, pp. 420-434, Feb. 2026.
    """

    def __init__(
        self,
        params: dict = None,
        crossover_operator=None,
        mutation_operator=None,
    ) -> None:
        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA2 (Tensorized).")

        super().__init__()

        self.crossover_operator = crossover_operator or SBXCrossoverTensor(env=Environment('numpy', 'float32'))
        self.mutation_operator = mutation_operator or PolynomialMutationTensor(env=Environment('numpy', 'float32'))
        self.build(params)

        self.DTYPE = None
        logger.info("Class overrided.")

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank) -> None:
        self._rank = rank

    @property
    def crowding_distance(self):
        return self._crowding_distance

    @crowding_distance.setter
    def crowding_distance(self, crowding_distance) -> None:
        self._crowding_distance = crowding_distance

    def compile(self, space: _MultiObjectiveTensorSpace) -> None:
        xp = space.env.xp
        self.DTYPE = space.env.dtype
        self.xp_module = xp

        n = space.n_agents
        self.n_agents = n

        self.d = space.X.shape[1]
        self.m = space.n_objectives

        self.rank = xp.zeros(n, dtype=xp.int32)
        self.crowding_distance = xp.zeros(n, dtype=self.DTYPE)

        self.P_tensor = xp.zeros((2 * n, self.d), dtype=self.DTYPE)
        self.F_tensor = xp.zeros((2 * n, self.m), dtype=self.DTYPE)

        self.P_tensor[:n] = xp.asarray(space.X, dtype=self.DTYPE)
        if space.F is not None:
            self.F_tensor[:n] = xp.asarray(space.F, dtype=self.DTYPE)

        self._LB = space.lb
        self._UB = space.ub

    def _dominance_matrix(self, F: Any, xp) -> Any:
        Fi = F[:, xp.newaxis, :]
        Fj = F[xp.newaxis, :, :]

        at_least_as_good = xp.all(Fi <= Fj, axis=2)
        strictly_better = xp.any(Fi < Fj, axis=2)

        return at_least_as_good & strictly_better

    def _nondominated_sort(self, F: Any, xp) -> Any:
        N = F.shape[0]

        D = self._dominance_matrix(F, xp)
        c = D.sum(axis=0).astype(xp.int32)

        r = xp.zeros(N, dtype=xp.int32)
        k = 0
        p = (c == 0)

        while bool(xp.any(p)):
            r = xp.where(p, k, r)

            p_int = p.astype(xp.int32)
            dominated_by_front = p_int @ D
            c = c - dominated_by_front - p_int

            k += 1
            p = (c == 0)

        return r

    def _compute_crowding_distance(self, costs, xp, mask=None):
        total_len = costs.shape[0]
        
        if mask is None:
            num_valid_elem = total_len
            mask = xp.ones(total_len, dtype=bool)
        else:
            num_valid_elem = int(mask.sum())
            
        if num_valid_elem == 0:
            return xp.full(total_len, -xp.inf)
            
        masked_costs = xp.where(mask[:, None], costs, xp.inf)
        
        rank = xp.argsort(masked_costs, axis=0)
        sorted_costs = xp.take_along_axis(costs, rank, axis=0)
        
        distance_range = sorted_costs[num_valid_elem - 1] - sorted_costs[0]
        distance_range = xp.where(distance_range == 0, 1e-9, distance_range)
        
        sorted_distances = xp.zeros_like(costs)
        
        if num_valid_elem > 2:
            interior_dists = (sorted_costs[2:num_valid_elem] - sorted_costs[:num_valid_elem - 2]) / distance_range
            sorted_distances[1:num_valid_elem - 1, :] = interior_dists
            
        sorted_distances[0, :] = xp.inf
        if num_valid_elem > 1:
            sorted_distances[num_valid_elem - 1, :] = xp.inf
            
        distance = xp.zeros_like(costs)
        xp.put_along_axis(distance, rank, sorted_distances, axis=0)
        
        crowding_distances = xp.where(mask[:, None], distance, -xp.inf)
        crowding_distances = xp.sum(crowding_distances, axis=1)
        
        return crowding_distances

    def _combined_rank_and_crowding(self, F: Any, xp):
        rank = self._nondominated_sort(F, xp)
        crowding = xp.zeros(F.shape[0], dtype=F.dtype)
        
        max_rank = int(xp.max(rank)) if F.shape[0] > 0 else 0
        for k in range(max_rank + 1):
            mask = (rank == k)
            front_cd = self._compute_crowding_distance(F, xp, mask=mask)
            crowding = xp.where(mask, front_cd, crowding)
            
        return rank, crowding

    def _tournament_selection_indices(self, n: int, xp) -> Any:
        i = xp.random.randint(0, n, size=n)
        j = (i + xp.random.randint(1, n, size=n)) % n

        ri, rj = self.rank[i], self.rank[j]
        ci, cj = self.crowding_distance[i], self.crowding_distance[j]

        prefer_i = (ri < rj) | ((ri == rj) & (ci > cj))
        return xp.where(prefer_i, i, j)

    def _environmental_selection(self, F_combined: Any, xp, n: int):
        rank, crowding = self._combined_rank_and_crowding(F_combined, xp)

        keys = xp.vstack((-crowding, rank))
        order = xp.lexsort(keys)
        best_idx = order[:n]

        return best_idx, rank, crowding

    def update(self, space: _MultiObjectiveTensorSpace, function) -> None:
        xp = space.env.xp
        n = self.n_agents

        sel_idx = self._tournament_selection_indices(n, xp)
        half = n // 2

        idx1 = sel_idx[:half]
        idx2 = sel_idx[half:half * 2]

        P1 = self.P_tensor[idx1]
        P2 = self.P_tensor[idx2]

        LB = self._LB[0]
        UB = self._UB[0]

        P_cross = self.crossover_operator(P1, P2, LB, UB)
        P_cross = xp.vstack(P_cross)

        P_children = self.mutation_operator(P_cross, LB, UB)

        num_children = P_children.shape[0]
        self.P_tensor[n: n + num_children] = P_children

        F_children = function(P_children, xp=xp)
        self.F_tensor[n: n + num_children] = F_children

        F_combined = self.F_tensor[: n + num_children]

        best_idx, rank_combined, crowding_combined = self._environmental_selection(
            F_combined, xp, n
        )

        self.P_tensor[:n] = self.P_tensor[best_idx].copy()
        self.F_tensor[:n] = self.F_tensor[best_idx].copy()

        space.X[:n] = self.P_tensor[:n]
        space.F = self.F_tensor[:n]

        self.rank = rank_combined[best_idx]
        self.crowding_distance = crowding_combined[best_idx]

    def evaluate(self, space: _MultiObjectiveTensorSpace, function) -> None:
        xp = space.env.xp

        F = function(self.P_tensor[:self.n_agents], xp=xp)
        self.F_tensor[:self.n_agents] = F
        space.F = self.F_tensor[:self.n_agents]

        self.rank, self.crowding_distance = self._combined_rank_and_crowding(
            self.F_tensor[:self.n_agents], xp
        )

        self.evaluate = lambda : None