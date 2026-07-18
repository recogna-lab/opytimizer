"""NSGA-II."""

from __future__ import annotations

import copy 

from dataclasses import dataclass
import numpy as np

from typing import List, Tuple, Optional, Any, Dict
import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer, Environment
from opytimizer.core.agent import Agent
from opytimizer.core.space import _MultiObjectiveSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation, SBXCrossoverTensor, PolynomialMutationTensor
from opytimizer.core.environment import Backend
from opytimizer.math.random import generate_integer_random_number 

logger = logging.get_logger(__name__)

class NSGA2:
    _registry = {}


    def __new__(cls, env: Optional[Environment] = None, **kwargs):
        if env is None: env = Environment().set_backend('cpu')
        if cls is NSGA2:
            target = cls._registry.get(env.backend)
            return super().__new__(target)
        return super().__new__(cls)

    def __init_subclass__(cls, backend: Backend = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if backend:
            key = backend.value if hasattr(backend, 'value') else backend
            NSGA2._registry[key] = cls
         

@dataclass
class _NSGA2Default(MultiObjectiveOptimizer, NSGA2, backend=Backend.CPU):
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
        **kwargs
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.

        """

        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA2 (Default).")

        super().__init__()

    
        self.crossover_operator = crossover_operator or SBXCrossover(return_mode='both')
        self.mutation_operator = mutation_operator or PolynomialMutation()

        self.build(params)

        self._isFirstIteration = True

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

    def _select_survivors(self, combined_population: List[Agent], space: _MultiObjectiveSpace) -> List[Agent]:
        """Selects the next generation of agents based on non-dominated sorting and crowding distance.

        Args:
            combined_population: Combined population of parents and offspring.
            space: Space containing agents and update-related information.
            function: Function to evaluate the fitness of the agents.

        Returns:
            (list): Selected agents for the next generation.

        """

        fronts, _ = self._fast_non_dominated_sort(combined_population)

        new_population = []
        n_agents = len(space.agents)

        # Assigns ranks and calculates crowding distance
        for front in fronts:
            if len(new_population) >= n_agents:
                break

            crowding = self._calculate_crowding_distance(front, combined_population)
            sorted_front = sorted(
                zip(front, crowding), key=lambda x: x[1], reverse=True
            )
            sorted_indices = [idx for idx, _ in sorted_front]

            remaining = n_agents - len(new_population)
            new_population.extend(
                [combined_population[i] for i in sorted_indices[:remaining]]
            )

        new_population = new_population[:n_agents]


        return new_population
    

    def update(self, space: _MultiObjectiveSpace, function) -> None:
        """Wraps NSGA-II over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        offspring = self._create_offspring(space)

        agent_shape = space.agents[0].position.shape
        n_offspring = min(len(offspring), len(space.agents))

        for i in range(n_offspring):
            offspring[i].position = offspring[i].position.reshape(agent_shape)

        for i in range(n_offspring):
            offspring[i].fit = function(offspring[i].position).squeeze()

        combined_population = space.agents + list(offspring)[:n_offspring]

        # Generates the offspring population (Q)
        new_pop = self._select_survivors(combined_population, space)

        # Updates the population with the new agents
        for i in range(len(space.agents)):
            space.agents[i] = new_pop[i]

    def evaluate(self, space, function):
        """Evaluates the fitness of the agents.

        Args:
            space: Space containing agents and evaluation-related information.
            function: Function to evaluate the fitness of the agents.

        """
        if self._isFirstIteration == True:
            for agent in space.agents:
                agent.fit = function(agent.position).reshape(-1)
            self._isFirstIteration = False

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




@dataclass
class _NSGA2Cuda(MultiObjectiveOptimizer, NSGA2, backend=Backend.CUDA):
    """Based on:
        Aguilar-Rivera, A. (2020). A GPU fully vectorized approach to accelerate performance of NSGA-2 based on stochastic non-domination sorting and grid-crowding.
        Applied Soft Computing, 88, 106047.
    """
    def __init__(
        self,
        params: dict = None,
        crossover_operator=None,
        mutation_operator=None,
        k: int = None,
        delta_n: int = 26,
        **kwargs
    ) -> None:
        """Initialization method.
 
        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.
 
        """
 
        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA2 (CUDA).")
 
        super().__init__()
 
        self.crossover_operator = crossover_operator or SBXCrossoverTensor(env=Environment('gpu', 'float32'))
        self.mutation_operator = mutation_operator or PolynomialMutationTensor(env=Environment('gpu', 'float32'))
        self.k = k
        self.delta_n = delta_n
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
 
    def compile(self, space: _MultiObjectiveSpace) -> None:
        xp = space.env.xp
        self.DTYPE = space.env.dtype
        self.rank = xp.zeros(space.n_agents)
        self.crowding_distance = xp.zeros(space.n_agents)
        if self.k is None:
            self.k = max(1, int(0.4 * space.n_agents))

        
        self.n_agents = space.n_agents
        sample_pos = xp.asarray(space.agents[0].position)
        self.agent_shape = sample_pos.shape
        self.d = sample_pos.size
        
       
        self.m = len(space.agents[0].fit) if space.agents[0].fit is not None else 2 
        
       
        self.P_tensor = xp.zeros((2 * self.n_agents, self.d), dtype=self.DTYPE)
        self.F_tensor = xp.zeros((2 * self.n_agents, self.m), dtype=self.DTYPE)

      
        for i, a in enumerate(space.agents):
            self.P_tensor[i] = xp.asarray(a.position, dtype=self.DTYPE).ravel()
            if a.fit is not None:
                self.F_tensor[i] = xp.asarray(a.fit, dtype=self.DTYPE).ravel()

        # cache bounds
        self._LB = xp.stack([xp.asarray(a.lb, dtype=self.DTYPE).ravel() for a in space.agents])
        self._UB = xp.stack([xp.asarray(a.ub, dtype=self.DTYPE).ravel() for a in space.agents])

        # Raw Kernel compilation
        import cupy as cp

        # Kernel 1: Stochastic Non-Domination Sorting
        self._nds_kernel = cp.ElementwiseKernel(
            in_params='raw T F, raw T Fs, int32 k, int32 m',
            out_params='int32 count',
            operation='''
                int n_idx = i; 
                int dom_count = 0;
                
                for (int j = 0; j < k; j++) {
                    bool j_dominates_i = true;
                    for (int l = 0; l < m; l++) {
                        
                        if (!(Fs[j * m + l] < F[n_idx * m + l])) {
                            j_dominates_i = false;
                            break; // Early exit
                        }
                    }
                    if (j_dominates_i) {
                        dom_count++;
                    }
                }
                count = dom_count;
            ''',
            name='stochastic_nds_kernel'
        )

        # Kernel 2: Grid-Based Crowding
        self._crowdin_kernel = cp.ElementwiseKernel(
            in_params='raw T Xg, raw T Xgs, int32 k, int32 m',
            out_params='int32 count',
            operation='''
                int n_idx = i;
                int match_count = 0;
                
                for (int j = 0; j < k; j++) {
                    bool same_cell = true;
                    for (int l = 0; l < m; l++) {
                        if (Xg[n_idx * m + l] != Xgs[j * m + l]) {
                            same_cell = false;
                            break; // Early exit 
                        }
                    }
                    if (same_cell) {
                        match_count++;
                    }
                }
                count = match_count;
            ''',
            name='grid_crowding_kernel'
        )

        # warm-up
        _dummy = xp.zeros((max(2, self.k + 1), 2))
        self._stochastic_nds(_dummy, xp)
        self._grid_crowding(_dummy, xp)
 
 
    def _stochastic_nds(self, F: Any, xp) -> Any:
        """Stochastic non-domination sorting - Eq. 3
 
           Instead of comparing all n individuals pairwise (O(n^2)), each individual
           is compared against a random sample of k individuals, reducing complexity
           to O(kn)
 
           Args:
             F: Objective matrix (n, m).
 
           Returns:
             Rank vector (n,): number of sampled individuals that dominate each x.
 
        """
 
        n, m = F.shape
        k = min(self.k, n)
 
        # Draw k individuals without replacement as the comparison batch
        idx = xp.random.choice(n, size=k, replace=False)
        Fs = F[idx]
 
        count = xp.zeros(n, dtype=xp.int32)
 
        self._nds_kernel(F, Fs, k, m, count)
 
        return count
 
    def _grid_crowding(self, F: Any, xp, x_min=None, x_max=None) -> Any:
        """Grid-based crowding density - Eq. 4-6
           Replaces the sequential crowding distance with a fully vectorized
           density metric: the search space is divided into a regular grid and
           each individual's density is estimated by counting how many sampled
           individuals share its cell. Higher value means more crowded region.
 
           Args:
             F: Objective matrix (n, m).
             x_min: Domain lower bounds per objective; defaults to F.min(axis=0).
             x_max: Domain upper bounds per objective; defaults to F.max(axis=0).
 
           Returns:
             Normalized density vector (n,).
 
        """
 
        n, m = F.shape
        k = min(self.k, n)
 
        eps = xp.finfo(xp.float64).tiny
 
        x_min = F.min(axis=0) if x_min is None else xp.asarray(x_min)
        x_max = F.max(axis=0) if x_max is None else xp.asarray(x_max)
 
        # Eq. 5: cell width per objective; guard against zero-range objectives
        delta = (x_max - x_min) / xp.array(self.delta_n - 1)
        delta = xp.where(delta == 0, xp.full_like(delta, eps), delta)
 
        # Eq. 4: map each individual to its integer grid cell
        Xg = xp.floor((F - x_min) / delta)  # (n, m)
 
        idx = xp.random.choice(n, size=k, replace=False)
 
        Xgs = Xg[idx]  # (k, m)
 
        # Eq. 6
 
        count = xp.zeros(n, dtype=xp.int32)
 
        self._crowdin_kernel(Xg, Xgs, k, m, count)
 
        # Normalization
        g_cwd = xp.maximum(count - 1, 0)
        return g_cwd / xp.array(n).clip(eps)
 
    def _combined_fitness(self, F: Any, xp, x_min=None, x_max=None) -> Any:
        # Lower value = better individual (low rank, low density)
        return self._stochastic_nds(F, xp) + self._grid_crowding(F, xp, x_min, x_max)
 
    def _tournament_selection_indices(self, n: int, xp) -> Any:
        i = xp.random.randint(0, n, size=n)
        j = (i + xp.random.randint(1, n, size=n)) % n
        ri, rj = self.rank[i], self.rank[j]
        ci, cj = self.crowding_distance[i], self.crowding_distance[j]
        prefer_i = (ri < rj) | ((ri == rj) & (ci < cj))
        return xp.where(prefer_i, i, j)
 
 
 
    def update(self, space: _MultiObjectiveSpace, function) -> None:
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
        self.P_tensor[n : n + num_children] = P_children

        F_children = function(P_children, xp=xp)
        self.F_tensor[n : n + num_children] = F_children

        F_combined = self.F_tensor[: n + num_children]
        
        best_idx = xp.argsort(self._combined_fitness(F_combined, xp))[:n]

        self.P_tensor[:n] = self.P_tensor[best_idx].copy()
        self.F_tensor[:n] = self.F_tensor[best_idx].copy()

        F_new = self.F_tensor[:n]
        self.rank = self._stochastic_nds(F_new, xp)
        self.crowding_distance = self._grid_crowding(F_new, xp)


        
 
    def evaluate(self, space: _MultiObjectiveSpace, function) -> None:
        xp = space.env.xp

        
        F = function(self.P_tensor[:self.n_agents], xp=xp)
        self.F_tensor[:self.n_agents] = F


        self.rank = self._stochastic_nds(self.F_tensor[:self.n_agents], xp)
        self.crowding_distance = self._grid_crowding(self.F_tensor[:self.n_agents], xp)

        self.evaluate = lambda: None



    def sync(self, space: _MultiObjectiveSpace):
        xp = space.env.xp
        X_cpu = self.P_tensor.tolist()
        F_cpu = self.F_tensor.tolist()

        for i, agent in enumerate(space.agents):
            agent.position[:] = xp.array(X_cpu[i]).reshape(agent.position.shape)
            agent.fit[:] = F_cpu[i]


        space.update_pareto_front()
