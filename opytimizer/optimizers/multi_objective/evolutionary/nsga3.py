"""NSGA-III"""

import numpy as np
from typing import List, Tuple

import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import _MultiObjectiveSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation
from opytimizer.utils.weights_vector import das_dennis

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
                n_divisions (default to 12). 
            
        """

        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA3.")

        super().__init__()

        self.crossover_operator = crossover_operator or SBXCrossover(return_mode="both", eta=30, gene_rate=1.0)
        self.mutation_operator = mutation_operator or PolynomialMutation()

       
        self._user_reference_points = reference_points
       

        # Internal state populated by compile() and updated each generation.
        self.reference_points: np.ndarray = None  # shape (H, M)
        self.rank: np.ndarray = None
        self._ideal_point: np.ndarray = None  # tracks global minimum per obj
        
        self.is_first_generation = True
        self.build(params)

        logger.info("Class overrided.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self, space: _MultiObjectiveSpace) -> None:
        """Compiles additional information used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        n_agents = space.n_agents

        # Infer number of objectives from the first agent's fitness vector.
        sample_fit = space.agents[0].fit
        if hasattr(sample_fit, "__len__"):
            n_objectives = len(sample_fit)
        else:
            raise e.ValueError(
                "NSGA-III requires multi-objective agents (agent.fit must be a vector)."
            )

        # Resolve reference points.
        if self._user_reference_points is not None:
            self.reference_points = np.array(self._user_reference_points, dtype=float)
        else:
           
            self.reference_points,_ = das_dennis(n_objectives, 12)
            logger.debug(
                f"Generated {len(self.reference_points)} Das-Dennis reference points "
                f"(M={n_objectives}, p={12})."
            )
            
            if len(self.reference_points) != space.n_agents:
                # Conflict: n_agents provided by user != Das-Dennis(N_OBJ, 12)
                raise e.ValueError('Error: conflict between the number of reference points and `n_agents` provided.')

        self.rank = np.zeros(n_agents, dtype=int)
        self._ideal_point = np.full(n_objectives, np.inf)

    # ------------------------------------------------------------------
    # Non-dominated sorting 
    # ------------------------------------------------------------------

    def _fast_non_dominated_sort(self, agents: list) -> List:
        """Performs the fast non-dominated sort.

        Args:
            agents: List of agents to be sorted.

        Returns:
            (list): List of fronts, each front being a list of agent indices.

        """

        n = len(agents)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if agents[i].dominates(agents[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif agents[j].dominates(agents[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

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

        # Assign ranks
        self.rank = np.zeros(n, dtype=int)
        for rank_val, front in enumerate(fronts):
            for idx in front:
                self.rank[idx] = rank_val

        return fronts

    # ------------------------------------------------------------------
    # Algorithm 2 — Adaptive normalisation
    # ------------------------------------------------------------------

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

        fitnesses = np.array([agents[i].fit for i in st_indices], dtype=float)  # (|St|, M)
        M = fitnesses.shape[1]

        # Update ideal point with the global minimum seen so far.
        self._ideal_point = np.minimum(self._ideal_point, fitnesses.min(axis=0))

        # Translate so that ideal point becomes the origin.
        f_prime = fitnesses - self._ideal_point  # (|St|, M)

        #  Extreme points via ASF
        # For objective i, weight vector w_i has w_i[i]=1 and w_i[j]=1e-6 for j≠i.
        extreme_indices = []
        for i in range(M):
            w = np.full(M, 1e-6)
            w[i] = 1.0
            # ASF(x, w) = max_j( f'_j(x) / w_j )
            asf_vals = np.max(f_prime / w, axis=1)
            extreme_indices.append(int(np.argmin(asf_vals)))

        extreme_points = f_prime[extreme_indices]  # (M, M)

        # Compute intercepts via the M-dimensional hyper-plane 

        try:
            b = np.linalg.solve(extreme_points, np.ones(M))
            intercepts = 1.0 / b
            # Guard against degenerate / non-positive intercepts.
            if np.any(intercepts <= 0) or np.any(np.isnan(intercepts)):
                raise np.linalg.LinAlgError("Non-positive intercepts")
        except np.linalg.LinAlgError:
            # Fallback: use the maximum value of each translated objective.
            intercepts = f_prime.max(axis=0)
            intercepts[intercepts == 0] = 1.0  # avoid division by zero

        # ---- Normalise ----
        f_n = f_prime / intercepts  # (|St|, M)

        return f_n

    # ------------------------------------------------------------------
    # Algorithm 3 — Association
    # ------------------------------------------------------------------

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
       

        # Unit reference lines: w_j = ref_points[j] / ||ref_points[j]||
        norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        w = ref_points / norms  # (H, M)

        # Vectorised perpendicular distance
        dot = f_n @ w.T  # (n_members, H)  ← s_i · w_j
        # Projection of each s_i onto each w_j
        proj = dot[:, :, np.newaxis] * w[np.newaxis, :, :]  # (n_members, H, M)
        diff = f_n[:, np.newaxis, :] - proj  # (n_members, H, M)
        dists = np.linalg.norm(diff, axis=2)  # (n_members, H)

        pi = np.argmin(dists, axis=1)   # (n_members,)
        d = dists[np.arange(n_members), pi]  # (n_members,)

        return pi, d

    # ------------------------------------------------------------------
    # Algorithm 4 — Niching (niche-preservation operation)
    # ------------------------------------------------------------------

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
        fl_remaining = list(fl_local)  # mutable copy
        rho = rho.copy()  # do not mutate the caller's array

        for _ in range(K):
            if not fl_remaining:
                break

            # Identify reference points with minimum niche count among those
            # that still have at least one candidate in fl_remaining.
            active_refs = set(pi[fl_remaining])
            min_rho = min(rho[j] for j in active_refs)
            j_min_set = [j for j in active_refs if rho[j] == min_rho]

            # Break ties randomly.
            j_bar = int(np.random.choice(j_min_set))

            # Candidates in fl_remaining associated with j_bar.
            candidates = [idx for idx in fl_remaining if pi[idx] == j_bar]

            if not candidates:
                # No member in Fl for this reference point → exclude it.
                continue

            if rho[j_bar] == 0:
                # Choose the candidate with the smallest perpendicular distance.
                chosen = candidates[int(np.argmin(d[candidates]))]
            else:
                # Choose a random candidate.
                chosen = int(np.random.choice(candidates))

            selected.append(chosen)
            fl_remaining.remove(chosen)
            rho[j_bar] += 1

        return selected

    # ------------------------------------------------------------------
    # Offspring generation
    # ------------------------------------------------------------------

    def _crossover(self, parent1: Agent, parent2: Agent) -> Tuple:
        return self.crossover_operator(parent1, parent2)

    def _mutation(self, agent: Agent) -> Agent:
        return self.mutation_operator(agent)

    def _create_offspring(self, space: _MultiObjectiveSpace) -> List:
        """Creates offspring via tournament selection."""

        # Tournament selection based on rank only 
        parents = self._tournament_selection(space.agents)
        offspring = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            children = self._crossover(p1, p2)
            for child in children:
                offspring.append(self._mutation(child))

        return offspring[: len(space.agents)]

    def _tournament_selection(self, agents: list) -> List:
        """Rank-based binary tournament selection."""
        selected = []
        for _ in range(len(agents)):
            i, j = np.random.choice(len(agents), 2, replace=False)
            winner = i if self.rank[i] <= self.rank[j] else j
            selected.append(agents[winner])
        return selected

    # ------------------------------------------------------------------
    # Survivor selection 
    # ------------------------------------------------------------------

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
 
        # Fill St front by front until |St| >= N 
        st_indices = []  # indices into `combined`
        last_front_idx = 0
        for fi, front in enumerate(fronts):
            if len(st_indices) + len(front) >= n_agents:
                last_front_idx = fi
                break
            st_indices.extend(front)
            last_front_idx = fi
        else:
            # All members fit exactly.
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
    # ------------------------------------------------------------------
    # Main update step
    # ------------------------------------------------------------------

    def update(self, space: _MultiObjectiveSpace, function) -> None:
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

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, space: _MultiObjectiveSpace, function) -> None:
        """Evaluates the fitness of all agents and updates the Pareto front.

        Args:
            space: Space containing agents and evaluation-related information.
            function: Objective function.

        """
        if self.is_first_generation == True:
            for agent in space.agents:
                agent.fit = function(agent.position)
            self.is_first_generation = False

        # Non-dominated sorting to assign initial ranks.
        _ = self._fast_non_dominated_sort(space.agents)

        # Initialise the running ideal point.
        fitnesses = np.array([a.fit for a in space.agents], dtype=float)
        self._ideal_point = np.minimum(self._ideal_point, fitnesses.min(axis=0))

        # Update the Pareto front (first non-dominated front).
        space.update_pareto_front(space.agents)