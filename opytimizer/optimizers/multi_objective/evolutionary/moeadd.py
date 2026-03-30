"""
MOEA/DD – Evolutionary Many-Objective Optimization Algorithm
           Based on Dominance and Decomposition.
"""

import numpy as np
from typing import Optional, Dict, Any, List

import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.core.function import Function
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation

logger = logging.get_logger(__name__)


class MOEA_DD(MultiObjectiveOptimizer):
    
    
    """
    
    MOEA_DD class, inherited from MultiObjectiveOptimizer.      
      
      References:
        K. Li, K. Deb, Q. Zhang and S. Kwong,
        "An Evolutionary Many-Objective Optimization Algorithm Based on
        Dominance and Decomposition,"
        IEEE Transactions on Evolutionary Computation, vol. 19, no. 5,
        pp. 694-716, Oct. 2015, doi: 10.1109/TEVC.2014.2373386.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        crossover_operator=None,
        mutation_operator=None,
        weight_vectors: np.ndarray = None,   
        T: int = 20,                          
        delta: float = 0.9,                  
        theta: float = 5.0,
    ):
        super().__init__()
        logger.info("Overriding class: MultiObjectiveOptimizer -> MOEA_DD.")

        self.crossover_operator = crossover_operator or SBXCrossover(
            rate=1.0, gene_rate=1.0, return_mode="both"
        )
        self.mutation_operator = mutation_operator or PolynomialMutation(
            rate=1.0 / 30.0
        )

        self.weight_vectors = weight_vectors  # (N, m)
        self.T = T
        self.delta = delta
        self.theta = theta

        
        self.z_star = None  # ideal point (m,)
        self.neighborhoods = None  # (N, T) indices into W
        self._subregion = None  # (N,) per current agent
        self._fronts = None  # non-dom level lists

        self.build(params)


    @property
    def T(self) -> int:
        return self._T

    @T.setter
    def T(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError('`T` should be an integer.')
        if value <= 0:
            raise e.ValueError('`T` should be higher than 0.')
        self._T = value

    @property
    def delta(self) -> float:
        return self._delta

    @delta.setter
    def delta(self, value) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError('`delta` should be a float.')
        if not (0.0 <= float(value) <= 1.0):
            raise e.ValueError('`delta` should be in [0, 1].')
        self._delta = float(value)

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError('`theta` should be a float.')
        if float(value) < 0.0:
            raise e.ValueError('`theta` should be >= 0.')
        self._theta = float(value)

   

    @staticmethod
    def _dominates(f1: np.ndarray, f2: np.ndarray) -> bool:
        """True iff f1 Pareto-dominates f2 (minimisation)."""
        return bool(np.all(f1 <= f2) and np.any(f1 < f2))

    # ------------------------------------------------------------------ #
    # PBI – Penalty-Based Boundary Intersection  [eqs. (2)-(4)]
    # ------------------------------------------------------------------ #

    def _gpbi_batch(
        self,
        fits: np.ndarray,   # (P, m) 
        weights:np.ndarray,   # (P, m)  
        z_star: np.ndarray,   # (m,)
    ) -> np.ndarray:           # (P,)
        """
        Vectorised g^pbi for P (fit, weight) pairs.

            d1 = ||(F(x) - z*)^T w|| / ||w||          [eq. (3)]
            d2 = ||F(x) - (z* + d1 · w/||w||)||       [eq. (4)]
            g^pbi = d1 + θ · d2                        [eq. (2)]
        """
        w_norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-12  # (P, 1)
        diff    = fits - z_star                                             # (P, m)
        d1      = np.sum(diff * weights, axis=1, keepdims=True) / w_norms  # (P, 1)
        proj    = z_star + d1 * (weights / w_norms)                        # (P, m)
        d2      = np.linalg.norm(fits - proj, axis=1)                      # (P,)
        return d1.ravel() + self._theta * d2                                # (P,)

    def _gpbi_single(
        self,
        fit:    np.ndarray,  # (m,)
        weight: np.ndarray,  # (m,)
        z_star: np.ndarray,  # (m,)
    ) -> float:
        """Scalar PBI – thin wrapper around the batch version."""
        return float(self._gpbi_batch(fit[None], weight[None], z_star)[0])

    # Convenience: compute g^pbi for every agent in `indices` against weight h.
    def _gpbi_for_indices(
        self,
        fits_p:  np.ndarray,  # full hybrid population (N+1, m)
        indices: np.ndarray,  # integer indices into fits_p
        h:       int,         # subregion / weight-vector index
        z_star:  np.ndarray,
    ) -> np.ndarray:           # (len(indices),)
        k = len(indices)
        return self._gpbi_batch(
            fits_p[indices],
            np.tile(self.weight_vectors[h], (k, 1)),
            z_star,
        )

    # ------------------------------------------------------------------ #
    # Subregion assignment – eq. (6)
    # ------------------------------------------------------------------ #

    def _assign_subregions(self, fits: np.ndarray) -> np.ndarray:
        """
        For each solution in `fits`, return the index of the weight vector
        forming the smallest acute angle with it (≡ maximum cosine).

        fits : (P, m)  →  (P,) integer array
        """
        W = self.weight_vectors                                     # (N, m)
        f_n = fits / (np.linalg.norm(fits, axis=1, keepdims=True) + 1e-12)
        w_n = W    / (np.linalg.norm(W,    axis=1, keepdims=True) + 1e-12)
        cos = f_n @ w_n.T                                             # (P, N)
        return np.argmax(cos, axis=1).astype(int)                     # (P,)

   

    def _fast_nondom_sort(self, fits: np.ndarray) -> List[List[int]]:
        P = len(fits)
        S = [[] for _ in range(P)]    # S[i]: indices dominated *by* i
        n = np.zeros(P, dtype=int)    # n[i]: how many solutions dominate i

        for i in range(P):
            for j in range(i + 1, P):
                if self._dominates(fits[i], fits[j]):
                    S[i].append(j);  n[j] += 1
                elif self._dominates(fits[j], fits[i]):
                    S[j].append(i);  n[i] += 1

        fronts: List[List[int]] = []
        current = list(np.where(n == 0)[0])
        while current:
            fronts.append(current)
            nxt: List[int] = []
            for i in current:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        nxt.append(j)
            current = nxt

        return fronts

    # ------------------------------------------------------------------ #
    # Tie-breaking helper – eq. (7):
    #   h = argmax_{i ∈ S} Σ_{x ∈ Φ^i} g^pbi(x | w^i, z*)
    # ------------------------------------------------------------------ #

    def _tiebreak_by_pbi_sum(
        self,
        candidates: np.ndarray,  # subregion index candidates
        fits_p:     np.ndarray,  # (N+1, m)
        sub_p:      np.ndarray,  # (N+1,)
        z_star:     np.ndarray,
    ) -> int:
        """Return the subregion index h ∈ candidates with the largest Σ g^pbi [eq. (7)]."""
        pbi_sums = np.array([
            self._gpbi_for_indices(
                fits_p, np.where(sub_p == h)[0], int(h), z_star
            ).sum()
            for h in candidates
        ])
        return int(candidates[np.argmax(pbi_sums)])

    # ------------------------------------------------------------------ #
    # Algorithm 5 – LOCATE_WORST
    # ------------------------------------------------------------------ #

    def _locate_worst(
        self,
        fits_p:   np.ndarray,        # (N+1, m)
        sub_p:    np.ndarray,        # (N+1,)
        fronts_p: List[List[int]],   # non-dom levels of hybrid population
        z_star:   np.ndarray,
    ) -> int:
        """
        Algorithm 5.
        Returns the index (into the hybrid population) of the solution to remove.

        Steps:
          1. Find the globally most crowded subregion h
             (tie → largest Σ g^pbi, eq. (7)).
          2. R ← solutions in Φ^h belonging to the worst non-dom level within h.
          3. Remove argmax_{x ∈ R} g^pbi(x | w^h, z*)   [eq. (9)].
        """
        N_w = len(self.weight_vectors)

        # --- Step 1: most crowded subregion globally --------------------
        niche = np.bincount(sub_p, minlength=N_w)   # (N_w,)
        max_n = niche.max()
        cands = np.where(niche == max_n)[0]

        h = (
            self._tiebreak_by_pbi_sum(cands, fits_p, sub_p, z_star)
            if cands.size > 1
            else int(cands[0])
        )

        # --- Step 2: R = Φ^h ∩ (worst non-domination level within h) ---
        in_h = np.where(sub_p == h)[0]                  # all indices in subregion h

        # map every solution to its front level
        front_of = np.zeros(len(fits_p), dtype=int)
        for lvl, front in enumerate(fronts_p):
            for idx in front:
                front_of[idx] = lvl

        worst_lvl = front_of[in_h].max()
        R = in_h[front_of[in_h] == worst_lvl]           # worst-level subset of Φ^h

        # --- Step 3: worst by g^pbi in R  [eq. (9)] --------------------
        pbi_R = self._gpbi_for_indices(fits_p, R, h, z_star)
        return int(R[np.argmax(pbi_R)])

    # ------------------------------------------------------------------ #
    # Algorithm 4 – UPDATE_POPULATION
    # ------------------------------------------------------------------ #

    def _update_population(
        self,
        agents: List[Agent],
        sub:    np.ndarray,    # (N,) current subregion per agent
        xc:     Agent,         # offspring solution
        xc_sub: int,           # offspring's subregion
        z_star: np.ndarray,
    ):
        """
        Algorithm 4 – steady-state population update.
        Returns (new_agents: List[Agent], new_sub: np.ndarray[int]).
        """
        N_w = len(self.weight_vectors)

        # Line 2 – P' ← P ∪ {xc}
        pop    = agents + [xc]
        sub_p  = np.append(sub, xc_sub)               # (N+1,)
        fits_p = np.array([a.fit for a in pop])        # (N+1, m)

        # Line 3 – update non-domination level structure of P'
        fronts_p = self._fast_nondom_sort(fits_p)
        l        = len(fronts_p)

        # ---- Decision tree (lines 4-25) --------------------------------
        if l == 1:
            # Line 5: all solutions are mutually non-dominated → LOCATE_WORST
            to_remove = self._locate_worst(fits_p, sub_p, fronts_p, z_star)

        else:
            last_front = fronts_p[-1]   # F_l (worst non-dom level)

            if len(last_front) == 1:
                # |F_l| = 1  ──────────────────────────────────────── lines 8-14
                xl_idx = last_front[0]
                xl_sub = int(sub_p[xl_idx])
                # |Φ^l|: how many solutions share xl's subregion (in P')
                if int((sub_p == xl_sub).sum()) > 1:
                    # Line 10: Φ^l not isolated → remove x^l directly
                    # (a better solution in the same subregion makes x^l redundant)
                    to_remove = xl_idx
                else:
                    # Lines 12-13: x^l is in an isolated subregion → second chance
                    # Preserve x^l; apply LOCATE_WORST on the whole P'
                    to_remove = self._locate_worst(fits_p, sub_p, fronts_p, z_star)

            else:
                # |F_l| > 1  ──────────────────────────────────────── lines 15-23
                last_arr  = np.array(last_front)    # indices in F_l inside pop
                last_subs = sub_p[last_arr]          # their subregion assignments

                # Line 16: most crowded h among *F_l* subregions
                #           (niche count = # F_l members per subregion)
                niche_fl = np.bincount(last_subs, minlength=N_w)
                max_n_fl = niche_fl.max()
                crowded  = np.where(niche_fl == max_n_fl)[0]

                # Tie-break eq. (7): Σ g^pbi over ALL solutions in P' for that subregion
                h = (
                    self._tiebreak_by_pbi_sum(crowded, fits_p, sub_p, z_star)
                    if crowded.size > 1
                    else int(crowded[0])
                )

                # |Φ^h|: total solutions in subregion h across ALL of P'
                niche_h_total = int((sub_p == h).sum())

                if niche_h_total > 1:
                    # Lines 17-19: Φ^h is not isolated →
                    #   remove the worst solution in Φ^h by g^pbi  [eq. (8)]
                    #   Note: eq. (8) ranges over ALL of Φ^h, not just F_l ∩ Φ^h.
                    in_h   = np.where(sub_p == h)[0]
                    pbi_h  = self._gpbi_for_indices(fits_p, in_h, h, z_star)
                    to_remove = int(in_h[np.argmax(pbi_h)])
                else:
                    # Lines 20-22: every F_l member is in an isolated subregion
                    # → second chance for all of them → LOCATE_WORST on full P'
                    to_remove = self._locate_worst(fits_p, sub_p, fronts_p, z_star)

        # Line 26 (conceptually) – remove chosen solution, return P
        keep       = [i for i in range(len(pop)) if i != to_remove]
        new_agents = [pop[i] for i in keep]
        new_sub    = sub_p[np.array(keep)]

        return new_agents, new_sub

    # ------------------------------------------------------------------ #
    # Algorithm 3 – MATING_SELECTION
    # ------------------------------------------------------------------ #

    def _mating_selection(
        self,
        i:      int,
        agents: List[Agent],
        sub:    np.ndarray,   # (N,)
    ):
        """
        Algorithm 3 – returns (parent1, parent2).

        With probability δ: restrict candidates to agents whose subregion
        is in the neighbourhood E(i) of weight vector i.
        Otherwise (or when the neighbourhood has < 2 members): random from P.
        """
        if np.random.random() < self._delta:
            # E(i) contains T weight-vector indices; solutions assigned to
            # any of those subregions are valid neighbourhood candidates.
            nbr_set  = set(self.neighborhoods[i].tolist())
            cand_idx = np.where(np.isin(sub, list(nbr_set)))[0]

            if cand_idx.size >= 2:
                chosen = np.random.choice(cand_idx, size=2, replace=False)
                return agents[int(chosen[0])], agents[int(chosen[1])]
            # Fall-through: neighbourhood underpopulated → use full population

        chosen = np.random.choice(len(agents), size=2, replace=False)
        return agents[int(chosen[0])], agents[int(chosen[1])]

    # ================================================================== #
    # opytimizer interface
    # ================================================================== #

    def compile(self, space: Space) -> None:
        """
        Algorithm 2 (partial) – neighbourhood construction.
        Weight vectors and initial population are user-provided;
        this method precomputes E(i) for every weight vector.
        """
        if len(self.weight_vectors) != space.n_agents:
            raise e.ValueError(
                'Number of `weight_vectors` must equal `n_agents`.'
            )

        W = self.weight_vectors                          # (N, m)

        # Vectorised pairwise Euclidean distance between weight vectors
        diff  = W[:, None, :] - W[None, :, :]           # (N, N, m)
        dists = np.linalg.norm(diff, axis=2)             # (N, N)

        # For each row, the T smallest distances give the T nearest neighbours.
        # (The weight vector itself is at distance 0 and is included in E(i)
        #  following the convention of MOEA/D / MOEA/DD.)
        self.neighborhoods = np.argsort(dists, axis=1)[:, :self.T]  # (N, T)

    def evaluate(self, space: Space, function: Function) -> None:
        """
        Algorithm 2 (partial) – initial evaluation.
        Evaluates the population, initialises z*, assigns subregions,
        and computes the first non-domination level structure.
        Called once before the main evolutionary loop.
        """
        fits = []
        for agent in space.agents:
            agent.fit = function(agent.position)
            fits.append(agent.fit)

        fits = np.array(fits)                            # (N, m)

        # Ideal point z* – initialised to component-wise minimum
        self.z_star = fits.min(axis=0)                  # (m,)

        # Subregion assignment for initial population  [eq. (6)]
        self._subregion = self._assign_subregions(fits)  # (N,)

        # Non-domination level structure
        self._fronts = self._fast_nondom_sort(fits)

        space.update_pareto_front(space.agents)
        

    def update(self, space: Space, function: Function) -> None:
        """
        Algorithm 1 (inner loop, lines 3-9) – one full generation.

        For each of the N weight vectors:
          1. Mating selection  (Algorithm 3)
          2. Crossover + mutation
          3. Evaluate offspring; update z*
          4. Steady-state population update  (Algorithm 4)
          5. Refresh non-domination structure for next iteration
        """
        N = len(space.agents)

        for i in range(N):

            # --- Step 1: mating selection (Algorithm 3) ----------------
            p1, p2 = self._mating_selection(i, space.agents, self._subregion)

            # --- Step 2: variation – crossover then polynomial mutation -
            offsprings = self.crossover_operator(parent1=p1, parent2=p2)
            xc = self.mutation_operator(offsprings[0])

            # --- Step 3: evaluate offspring; update ideal point z* -----
            xc.fit    = function(xc.position)
            self.z_star = np.minimum(xc.fit, self.z_star)

            # --- Subregion of offspring  [eq. (6)] ---------------------
            xc_sub = int(self._assign_subregions(xc.fit[None])[0])

            # --- Step 4: steady-state update (Algorithm 4) -------------
            space.agents, self._subregion = self._update_population(
                space.agents,
                self._subregion,
                xc,
                xc_sub,
                self.z_star,
            )

            # --- Step 5: refresh front structure for the next iteration -
            # (In the paper this is the incremental update of [66];
            #  here we recompute from scratch – functionally identical.)
            fits = np.array([a.fit for a in space.agents])
            self._fronts = self._fast_nondom_sort(fits)
            
           

       