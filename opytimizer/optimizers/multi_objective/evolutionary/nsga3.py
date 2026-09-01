import numpy as np
from typing import List, Tuple

import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer, Function
from opytimizer.core.agent import Agent
from opytimizer.core.space import _MultiObjectiveSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation
from opytimizer.utils.reference_vectors import das_dennis

logger = logging.get_logger(__name__)

class NSGA3(MultiObjectiveOptimizer):
    """NSGA3 class, inherited from MultiObjectiveOptimizer.

    Replaces the crowding distance operator of NSGA-II with a reference-point-based
    niching strategy, making it effective for problems with four or more objectives.

    References:
        K. Deb and H. Jain. An Evolutionary Many-Objective Optimization Algorithm
        Using Reference-Point-Based Nondominated Sorting Approach, Part I.
        IEEE Transactions on Evolutionary Computation (2014).

    """

    def __init__(
        self,
        params: dict = None,
        crossover_operator=None,
        mutation_operator=None,
        reference_points: np.ndarray = None,
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.
            reference_points: Pre-computed reference points of shape (H, M).
                If None, they are generated via Das and Dennis's approach using
                n_divisions (default set to 12). 
            
        """

        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA3.")

        super().__init__()

        self.crossover_operator = crossover_operator or SBXCrossover(n_offspring=2, eta=30, gene_rate=1.0)
        self.mutation_operator = mutation_operator or PolynomialMutation()

        self._user_reference_points = reference_points

        self.reference_points: np.ndarray = None 
        self.rank: np.ndarray = None
        
        self.is_first_generation = True
        self.build(params)

        logger.info("Class overrided.")

    @property
    def rank(self) -> np.ndarray:
        """Array of non-domination ranks."""
        return self._rank

    @rank.setter
    def rank(self, rank: np.ndarray) -> None:
        if rank is not None and not isinstance(rank, np.ndarray):
            raise e.TypeError("`rank` should be a numpy array")
        self._rank = rank

    @property
    def reference_points(self) -> np.ndarray:
        """Reference points on the normalised hyper-plane, shape (H, M)."""
        return self._reference_points

    @reference_points.setter
    def reference_points(self, ref: np.ndarray) -> None:
        if ref is not None and not isinstance(ref, np.ndarray):
            raise e.TypeError("`reference_points` should be a numpy array")
        self._reference_points = ref

    def compile(self, space: _MultiObjectiveSpace) -> None:
        """Compiles additional information used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        n_agents = space.n_agents

        sample_fit = space.agents[0].fit
        if hasattr(sample_fit, "__len__"):
            n_objectives = len(sample_fit)
        else:
            raise e.ValueError(
                "NSGA-III requires multi-objective agents (agent.fit must be a vector)."
            )

        if self._user_reference_points is not None:
            self.reference_points = np.asarray(self._user_reference_points, dtype=float)
        else:
            self.reference_points,_ = das_dennis(n_objectives, 12)
            logger.debug(
                f"Generated {len(self.reference_points)} Das-Dennis reference points "
                f"(M={n_objectives}, p={12})."
            )
            
            if len(self.reference_points) != space.n_agents:
                raise e.ValueError('Error: conflict between the number of reference points and `n_agents` provided.')

        self.rank = np.zeros(n_agents, dtype=int)

    def _fast_non_dominated_sort(self, agents: list) -> List:
        """Performs the fast non-dominated sort.

        Args:
            agents: List of agents to be sorted.

        Returns:
            (list): List of fronts, each front being a list of agent indices.

        """
        n = len(agents)
        fits = np.array([a.fit for a in agents])
        
        fits_i = fits[:, np.newaxis, :]
        fits_j = fits[np.newaxis, :, :]
        
        dominates = np.logical_and(np.all(fits_i <= fits_j, axis=2), np.any(fits_i < fits_j, axis=2))
        
        domination_count = np.sum(dominates, axis=0)
        dominated_solutions = [np.where(dominates[i])[0].tolist() for i in range(n)]
        
        fronts = [np.where(domination_count == 0)[0].tolist()]
        
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

        self.rank = np.zeros(n, dtype=int)
        for rank_val, front in enumerate(fronts):
            for idx in front:
                self.rank[idx] = rank_val

        return fronts

    def _normalize(self, agents: list, st_indices: list) -> np.ndarray:
        """Adaptively normalises objective values of St members (Algorithm 2).

        Updates the running ideal point, translates objectives, locates extreme
        points via the Achievement Scalarizing Function (ASF), computes hyper-plane
        intercepts, and returns the normalised fitness matrix for all agents in St.

        Args:
            agents: Full combined population (parents + offspring).
            st_indices: Indices (into agents) of all members of St.

        Returns:
            (np.ndarray): Normalised objective matrix of shape (|St|, M).

        """

        fitnesses = np.array([agents[i].fit for i in st_indices], dtype=float)
        M = fitnesses.shape[1]

        ideal_point = fitnesses.min(axis=0)

        f_prime = fitnesses - ideal_point 

        extreme_indices = []
        for i in range(M):
            w = np.full(M, 1e-6)
            w[i] = 1.0
            asf_vals = np.max(f_prime / w, axis=1)
            extreme_indices.append(int(np.argmin(asf_vals)))

        extreme_points = f_prime[extreme_indices] 

        try:
            b = np.linalg.solve(extreme_points, np.ones(M))
            intercepts = 1.0 / b
            if np.any(intercepts <= 0) or np.any(np.isnan(intercepts)):
                raise np.linalg.LinAlgError()
        except np.linalg.LinAlgError:
            intercepts = np.max(f_prime, axis=0)
            intercepts[intercepts == 0] = 1e-6

        f_n = f_prime / intercepts 

        return f_n

    def _associate(
        self, f_n: np.ndarray, ref_points: np.ndarray
    ) -> Tuple:
        """Associates each population member with its closest reference line.

        The reference line for reference point z is the ray from the origin
        through z. The perpendicular distance from point s to reference line
        w = z / ||z|| is:

            d_perp(s, w) = || s - (s·w / w·w) * w ||

        Args:
            f_n: Normalised objective matrix, shape (n_members, M).
            ref_points: Reference points, shape (H, M).

        Returns:
            (tuple): pi (np.ndarray of shape (n_members,)) with the index of
                the closest reference point, and d (np.ndarray of shape
                (n_members,)) with the corresponding perpendicular distances.

        """

        n_members = f_n.shape[0]

        norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        w = ref_points / norms 

        dot = f_n @ w.T
        s_norm_sq = np.sum(f_n**2, axis=1)[:, np.newaxis]
        dists = np.sqrt(np.maximum(0, s_norm_sq - dot**2))

        pi = np.argmin(dists, axis=1) 
        d = dists[np.arange(n_members), pi] 

        return pi, d

    def _niching(
        self,
        K: int,
        rho: np.ndarray,
        pi: np.ndarray,
        d: np.ndarray,
        fl_local: list,
    ) -> List:
        """Chooses K members from the last front Fl using niche preservation.

        Args:
            K: Number of members to select.
            rho: Niche counts of reference points, shape (H,), counting members
                already added to Pt+1 (i.e. from fronts 1 … l-1 mapped into
                the St index space).
            pi: Association array for all St members, shape (|St|,).
            d: Distance array for all St members, shape (|St|,).
            fl_local: Indices *within the St index space* of members in Fl.

        Returns:
            (list): Selected indices (in St index space) from fl_local.

        """

        selected = []
        rho = rho.copy() 
        
        candidates_map = {j: [] for j in range(len(rho))}
        for idx in fl_local:
            candidates_map[pi[idx]].append(idx)
            
        active_refs = {j for j, cands in candidates_map.items() if cands}

        for _ in range(K):
            if not active_refs:
                break

            min_rho = min(rho[j] for j in active_refs)
            j_min_set = [j for j in active_refs if rho[j] == min_rho]

            j_bar = int(np.random.choice(j_min_set))

            candidates = candidates_map[j_bar]

            if rho[j_bar] == 0:
                chosen_idx = int(np.argmin(d[candidates]))
                chosen = candidates[chosen_idx]
            else:
                chosen_idx = int(np.random.choice(len(candidates)))
                chosen = candidates[chosen_idx]

            selected.append(chosen)
            candidates.pop(chosen_idx)
            rho[j_bar] += 1
            
            if not candidates:
                active_refs.remove(j_bar)

        return selected

    def _crossover(self, parent1: Agent, parent2: Agent) -> Tuple:
        return self.crossover_operator(parent1, parent2)

    def _mutation(self, agent: Agent) -> Agent:
        return self.mutation_operator(agent)

    def _create_offspring(self, space: _MultiObjectiveSpace) -> List:
        """Creates offspring via tournament selection."""
        parents = self._tournament_selection(space.agents)
        offspring = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            children = self._crossover(p1, p2)
            for child in children:
                offspring.extend(self._mutation(child))
        
        return offspring[: len(space.agents)]

    def _tournament_selection(self, agents: list) -> List:
        """Rank-based binary tournament selection."""
        selected = []
        n_agents = len(agents)
        for _ in range(n_agents):
            selected.append(agents[np.random.choice(n_agents)])
        return selected

    def _select_survivors(self, combined: list, n_agents: int) -> List:
        """Selects the next generation using non-dominated sorting + niching.
 
        Implements the main body of Algorithm 1 from the paper.
 
        Args:
            combined: Combined population (parents + offspring).
            n_agents: Target population size N.
 
        Returns:
            (list): Selected agents for the next generation (length n_agents).
 
        """
 
        fronts = self._fast_non_dominated_sort(combined)
 
        st_indices = []
        last_front_idx = 0
        for fi, front in enumerate(fronts):
            if len(st_indices) + len(front) >= n_agents:
                last_front_idx = fi
                break
            st_indices.extend(front)
            last_front_idx = fi
        else:
            return [combined[i] for i in st_indices]
 
        fl = fronts[last_front_idx]
        K = n_agents - len(st_indices)
 
        full_st = st_indices + list(fl)
 
        f_n = self._normalize(combined, full_st)
 
        fl_local = list(range(len(st_indices), len(full_st)))
 
        pi, d = self._associate(f_n, self.reference_points)
 
        H = len(self.reference_points)
        rho = np.zeros(H, dtype=int)
        for local_idx in range(len(st_indices)):
            rho[pi[local_idx]] += 1
 
        chosen_local = self._niching(K, rho, pi, d, fl_local)
 
        final_indices = st_indices + [full_st[loc] for loc in chosen_local]
        return [combined[i] for i in final_indices[:n_agents]]

    def update(self, space: _MultiObjectiveSpace, function: Function) -> None:
        """Wraps NSGA-III over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: Objective function used to evaluate offspring.

        """

        offspring = self._create_offspring(space)
        
        for i in range(len(offspring)):
            offspring[i].fit = function(offspring[i].position)

        combined = space.agents + offspring
        new_pop = self._select_survivors(combined, len(space.agents))

        for i in range(len(space.agents)):
            space.agents[i] = new_pop[i]

        _ = self._fast_non_dominated_sort(space.agents)

    def evaluate(self, space: _MultiObjectiveSpace, function: Function):
        super().evaluate(space, function)
        self.evaluate = lambda : None
    