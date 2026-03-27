"""
    Knee Point Driven Evolutionary Algorithm (KnEA)
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Union

from opytimizer.core import Agent, Function, MultiObjectiveOptimizer
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation
import opytimizer.math.random as opt_r
import opytimizer.utils.exception as e

class KnEA(MultiObjectiveOptimizer):
    """
        Reference:
            Zhang, X., Tian, Y., & Jin, Y. (2014).
            A knee point-driven evolutionary algorithm for many-objective optimization.
            IEEE Transactions on Evolutionary Computation, 19(6), 761-776.
    """

    def __init__(
        self,
        params: dict = None,
        crossover_operator=None,
        mutation_operator=None,
        k: int = 3,
        T: float = 0.5,
    ):
        super().__init__()
        self.is_first_generation = True
        self.r = None
        self.t = None
        self.knn_num = k
        self.K = None
        self.T = T

        self.crossover_operator = crossover_operator or SBXCrossover(return_mode="both")
        self.mutation_operator = mutation_operator or PolynomialMutation(rate=1 / 30)

        
        self._obj_matrix = None # (2N, M) fitness matrix
        self._weighted_dists = None  # (2N,)  pre-computed weighted distances

        self.build(params)
        
    @property
    def knn_num(self) -> int:
        return self._knn_num
    @knn_num.setter
    def knn_num(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError('`k` should be an integer.')
        if value <= 0:
            raise e.ValueError('`k` should be higher than 0.')
        
        self._knn_num = value
        
        
    @property
    def T(self) -> float:
        return self._T
    @T.setter
    def T(self, value: int) -> None:
        if not isinstance(value, float):
            raise e.TypeError('`T` should be a float.')
        if value <= 0. or value > 1.0:
            raise e.ValueError('`T` should be within (0, 1] interval.')
        
        self._T= value
        
        

    def compile(self, space):
        self.r = -np.ones(space.n_agents * 2) 
        self.t = -np.ones(space.n_agents * 2)
        self.K = []

    # ------------------------------------------------------------------
    # Algorithm 2 – Mating Selection
    # ------------------------------------------------------------------
    def _mating_selection(self, P: List[Agent], K: List[Agent], N: int) -> List[Agent]:
        """Binary tournament selection using dominance > knee > weighted distance."""

        # weighted distance in objective space
        self._precompute_weighted_distances(P)

        K_set = set(id(a) for a in K)   # O(1) membership test by identity

        Q: List[Agent] = []
        indices = list(range(len(P)))

        while len(Q) < N:
            ia, ib = np.random.choice(indices, 2, replace=False)
            a, b = P[ia], P[ib]

            if a.dominates(b):
                Q.append(a)
            elif b.dominates(a):
                Q.append(b)
            else:
                a_is_knee = id(a) in K_set
                b_is_knee = id(b) in K_set

                if a_is_knee and not b_is_knee:
                    Q.append(a)
                elif b_is_knee and not a_is_knee:
                    Q.append(b)
                else:
                    dw_a = self._weighted_dists[ia]
                    dw_b = self._weighted_dists[ib]
                    if dw_a > dw_b:
                        Q.append(a)
                    elif dw_b > dw_a:
                        Q.append(b)
                    else:
                        Q.append(a if opt_r.generate_uniform_random_number() < 0.5 else b)

        return Q

    def _precompute_weighted_distances(self, P: List[Agent]) -> None:
        """
        Pre-compute weighted distances for ALL agents in P at once.

        Equations (1)-(3) from the paper.
        """
        # (N, M) — objective space
        obj = np.array([a.fit for a in P], dtype=float)

        # Pairwise Euclidean distances in objective space  (N, N)
        D = cdist(obj, obj, metric="euclidean")
        np.fill_diagonal(D, np.inf)

        N = len(P)
        k = self.knn_num
        weighted_dists = np.empty(N, dtype=float)

        
        knn_idx = np.argpartition(D, k, axis=1)[:, :k]    
        knn_d = D[np.arange(N)[:, None], knn_idx]       

        mean_d = knn_d.mean(axis=1, keepdims=True)          

        # Eq. (3): r_pi = 1 / |dis_{p,pi} - mean| 
        diff = np.abs(knn_d - mean_d)
        diff = np.where(diff < 1e-10, 1e-10, diff)
        r = 1.0 / diff                                        

        # Eq. (2): w_pi = r_pi / sum_j r_pj
        w = r / r.sum(axis=1, keepdims=True)                

        # Eq. (1): DW(p) = sum_i w_pi * dis_{p,pi}
        weighted_dists = (w * knn_d).sum(axis=1)                 

        self._weighted_dists = weighted_dists

  
    def _genetic_operators(
        self, mating: List[Agent], N: int, function: Function
    ) -> List[Agent]:
        parents_indices = np.random.randint(0, N, size=(N // 2, 2))
        offsprings_list: List[Agent] = []

        for p1_idx, p2_idx in parents_indices:
            offsprings = self.crossover_operator(
                parent1=mating[p1_idx], parent2=mating[p2_idx]
            )
            for i in range(len(offsprings)):
                offsprings[i] = self.mutation_operator(offsprings[i])
                offsprings[i].fit = function(offsprings[i].position)
            offsprings_list.extend(offsprings)

        return offsprings_list

    
    def _fast_non_dominated_sort(self, agents: List[Agent]) -> List[List[int]]:
     
        fits = np.array([a.fit for a in agents], dtype=float)  # (N, M)
        N = len(agents)

         
        fi = fits[:, None, :] # (N, 1, M)
        fj = fits[None, :, :] # (1, N, M)

        all_leq = np.all(fi <= fj, axis=2) # (N, N) — i <= j in every objective
        any_lt = np.any(fi <  fj, axis=2) # (N, N) — i <  j in at least one
        dom_mat = all_leq & any_lt  # (N, N) — i dominates j

        # Remove self-domination (diagonal)
        np.fill_diagonal(dom_mat, False)

        domination_count = dom_mat.sum(axis=0).astype(int)        
        dominated_solutions = [list(np.where(dom_mat[i])[0]) for i in range(N)]

        fronts = [list(np.where(domination_count == 0)[0])]
        self.rank = np.empty(N, dtype=int)
        for idx in fronts[0]:
            self.rank[idx] = 0

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: List[int] = []
            for j in fronts[i]:
                for k in dominated_solutions[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            if next_front:
                for idx in next_front:
                    self.rank[idx] = i + 1
                fronts.append(next_front)
            i += 1

        return fronts

    # ------------------------------------------------------------------
    # Algorithm 3 – Finding Knee Points
    # ------------------------------------------------------------------
    def _finding_knee_point(
        self, current_population: List[Agent], F: List[List[int]]
    ):
      
        pop_objs = np.array([ind.fit for ind in current_population], dtype=float)
        M = pop_objs.shape[1]

        all_knee_indices:  List[np.ndarray] = []
        all_sorted_fronts: List[np.ndarray] = []
        front_map:  List[int] = []  

        for fi_idx, front_indices in enumerate(F):
            if len(front_indices) == 0:
                continue

            front_indices = np.asarray(front_indices)
            front_values = pop_objs[front_indices]   
            n_points = len(front_indices)

            # Extreme solutions 
            local_extreme_idxs = np.argmax(front_values, axis=0)
            unique_extreme_local_idxs = np.unique(local_extreme_idxs)
            extreme_points = front_values[unique_extreme_local_idxs]

            # Update r ( Eq. 7) 
            if self.t[fi_idx] == -1:
                self.r[fi_idx] = 1.0
            else:
                self.r[fi_idx] = self.r[fi_idx] / np.exp((1.0 - self.t[fi_idx] / self.T) / M)

            # fmax / fmin
            fmax = front_values.max(axis=0)
            fmin = front_values.min(axis=0)

            # Neighbourhood half-widths R (Eq. 6) 
            R = (fmax - fmin) * self.r[fi_idx] # (M,)

            # Degenerate case: treat all as knees 
            if len(unique_extreme_local_idxs) < M or n_points <= M:
                all_knee_indices.append(front_indices)
                all_sorted_fronts.append(front_indices)
                front_map.append(fi_idx)
                self.t[fi_idx] = 1.0
                continue

            # hyperplane L:  extreme_pts · w = 1  ( Eq. 4/5)
            try:
                b = np.ones(len(unique_extreme_local_idxs))
                w, _, _, _ = np.linalg.lstsq(extreme_points, b, rcond=None)
                norm_w = np.linalg.norm(w)
                if norm_w < 1e-10:
                    raise np.linalg.LinAlgError("zero-norm hyperplane normal")
            except np.linalg.LinAlgError:
                all_knee_indices.append(front_indices)
                all_sorted_fronts.append(front_indices)
                front_map.append(fi_idx)
                self.t[fi_idx] = 1.0
                continue

            # --- Signed distance to L (Eq. 5)
            residuals = np.dot(front_values, w) - 1.0  
            abs_dist  = np.abs(residuals) / norm_w
            signed_distances = np.where(residuals < 0, abs_dist, -abs_dist)

            # --- Sort descending 
            sorted_local_idx = np.argsort(-signed_distances)       
            global_sorted_front = front_indices[sorted_local_idx]

            # --- Greedy knee sweep
            size_fi  = n_points
            remaining = np.ones(n_points, dtype=bool)     
            knee_local: List[int] = []

            
           
            for p_local in sorted_local_idx:
                if not remaining[p_local]:
                    continue

                knee_local.append(p_local)

                # NB = { a | |f^j_a - f^j_p| <= R^j  for all j } 
                diff  = np.abs(front_values - front_values[p_local])  # (n, M)
                in_nb = np.all(diff <= R, axis=1) # (n,)
                remaining[in_nb] = False # Fi ← Fi \ NB

            # --- t = |K| / SizeFi 
            self.t[fi_idx] = len(knee_local) / size_fi

            all_knee_indices.append(front_indices[knee_local])
            all_sorted_fronts.append(global_sorted_front)
            front_map.append(fi_idx)

        return all_knee_indices, all_sorted_fronts, front_map

    # ------------------------------------------------------------------
    # Algorithm 4 – Environmental Selection
    # ------------------------------------------------------------------
    def _environmental_selection(
        self,
        current_population: List[Agent],
        F: List[List[int]],
        K: List[np.ndarray],    
        sorted_fronts: List[np.ndarray],
        front_map: List[int],  
        N: int,
    ) -> List[Agent]:
       
        Q: List[Agent] = []

        # Build a lookup: F index → position in K/sorted_fronts
        fi_to_k = {fi: ki for ki, fi in enumerate(front_map)}

        critical_fi_idx = None  # index in F of the critical front

        for fi_idx, front in enumerate(F):
            if len(Q) + len(front) <= N:
                for idx in front:
                    Q.append(current_population[idx])
                if len(Q) == N:
                    return Q
            else:
                critical_fi_idx = fi_idx
                break

        # All fronts fit — no critical front needed
        if critical_fi_idx is None:
            return Q

       
        ki = fi_to_k.get(critical_fi_idx, None)
        if ki is None:
            
            remaining_slots = N - len(Q)
            for idx in F[critical_fi_idx][:remaining_slots]:
                Q.append(current_population[idx])
            return Q

        knee_global_idxs = set(int(i) for i in K[ki])
        sorted_front_idxs = list(sorted_fronts[ki])   

        # Knee agents from critical front
        knee_agents_ordered = [
            current_population[idx]
            for idx in sorted_front_idxs
            if idx in knee_global_idxs
        ]   
        remaining_slots = N - len(Q)
        n_knees = len(knee_agents_ordered)

        if n_knees <= remaining_slots:
           
            Q.extend(knee_agents_ordered)
            remaining_slots -= n_knees

            if remaining_slots > 0:
                non_knee_agents = [
                    current_population[idx]
                    for idx in sorted_front_idxs
                    if idx not in knee_global_idxs
                ]   #
                Q.extend(non_knee_agents[:remaining_slots])
        else:
            # |Q ∪ knees| > N  -> keep only the knees with LARGEST distance
            Q.extend(knee_agents_ordered[:remaining_slots])

        return Q

    # ------------------------------------------------------------------
    # Evaluate / Update
    # ------------------------------------------------------------------
    def evaluate(self, space, function):
        if self.is_first_generation == True:
            super().evaluate(space, function)
            self.is_first_generation = False
            
        else:
            space.update_pareto_front(space.agents)
            
       
    def update(self, space, function):
        current_population = space.agents.copy()

        # Mating + offspring
        mating = self._mating_selection(P=space.agents, K=self.K, N=space.n_agents)
        offspring = self._genetic_operators(mating, space.n_agents, function)
        current_population.extend(offspring)

        # Non-dominated sort on combined population (2N)
        fronts = self._fast_non_dominated_sort(current_population)

        # Knee point detection
        knee_indices, sorted_fronts, front_map = self._finding_knee_point(
            current_population, fronts
        )

        # Environmental selection -> next generation
        space.agents = self._environmental_selection(
            current_population, fronts, knee_indices, sorted_fronts, front_map,
            space.n_agents,
        )

        
        surviving_ids = set(id(a) for a in space.agents)
        flat_knee_idxs = [int(idx) for front in knee_indices for idx in front]
        self.K = [
            current_population[idx]
            for idx in flat_knee_idxs
            if id(current_population[idx]) in surviving_ids
        ]
        
       