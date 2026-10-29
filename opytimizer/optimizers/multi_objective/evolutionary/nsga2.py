"""NSGA-II."""

from __future__ import annotations

import functools
from typing import Any, Dict, List

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import (
    Environment,
    MultiObjectiveOptimizer,
    TensorizedMultiObjectiveOptimizer,
)
from opytimizer.core.agent import Agent
from opytimizer.core.space import _MultiObjectiveSpace, _MultiObjectiveTensorSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import (
    PolynomialMutation,
    PolynomialMutationTensor,
    SBXCrossover,
    SBXCrossoverTensor,
)

logger = logging.get_logger(__name__)

# Upper bound on the number of booleans held by each temporary of the blocked
# dominance computation
_DOMINANCE_BLOCK_ELEMENTS = 1 << 26

# Number of non-dominated fronts peeled between two device -> host emptiness checks.
_PEEL_BATCH_CUDA = 4


@functools.lru_cache(maxsize=None)
def _get_dominance_kernel(xp):
    """Fused CUDA kernel computing the full (N, N) Pareto-dominance matrix.

    out[a, b] is True iff F[a] dominates F[b] (no worse in every objective and
    strictly better in at least one). Each thread scans the M objectives of one
    pair and exits as soon as it finds an objective where `a` is worse, so no
    (N, N, M) intermediate is ever written to memory.
    """

    return xp.ElementwiseKernel(
        "int32 ia, int32 ib, raw T F, int32 M",
        "bool dom",
        """
        bool no_worse = true;
        bool better = false;
        for (int k = 0; k < M; ++k) {
            T va = F[ia * M + k];
            T vb = F[ib * M + k];
            if (va > vb) { no_worse = false; break; }
            if (va < vb) { better = true; }
        }
        dom = no_worse && better;
        """,
        "nsga2_dominance",
    )


class NSGA2(MultiObjectiveOptimizer):
    """NSGA2 class, inherited from MultiObjectiveOptimizer.

    References:
        K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
        multiobjective genetic algorithm: NSGA-II," IEEE Trans. Evol. Comput.,
        vol. 6, no. 2, pp. 182-197, 2002.

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
            crossover_operator: Crossover operator applied to pairs of parents.
            mutation_operator: Mutation operator applied to every child.

        """

        super().__init__()
        self.crossover_operator = crossover_operator or SBXCrossover()
        self.mutation_operator = mutation_operator or PolynomialMutation()
        self.build(params)

    @property
    def rank(self) -> np.ndarray:
        """Non-dominated front index of each agent (0 is the best front)."""
        return self._rank

    @rank.setter
    def rank(self, rank: np.ndarray) -> None:
        self._rank = np.asarray(rank)

    @property
    def crowding_distance(self) -> np.ndarray:
        """Crowding distance of each agent inside its own front."""
        return self._crowding_distance

    @crowding_distance.setter
    def crowding_distance(self, crowding_distance: np.ndarray) -> None:
        self._crowding_distance = np.asarray(crowding_distance)

    def compile(self, space: _MultiObjectiveSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A MultiObjectiveSpace object.

        """

        self.crowding_distance = np.zeros(space.n_agents)

    def _fast_non_dominated_sort(self, agents: List[Agent]):
        """Sorts the agents into non-dominated fronts.

        Args:
            agents: List of agents to be sorted.

        Returns:
            (Tuple[List[List[int]], np.ndarray]): Indices of the agents of each front
                (best front first) and the front index of every agent.

        """

        n = len(agents)
        population_fitness = np.array([ag.fit for ag in agents])

        P = population_fitness[:, np.newaxis, :]  # (N, 1, M)
        Q = population_fitness[np.newaxis, :, :]  # (1, N, M)

        # dominance_matrix[i, j] is True iff agent i dominates agent j
        cond_1 = np.all(P <= Q, axis=2)
        cond_2 = np.any(P < Q, axis=2)
        dominance_matrix = cond_1 & cond_2

        # Number of agents dominating each agent (0 means it is in the first front)
        num_dominance = np.sum(dominance_matrix, axis=0)
        local_rank = np.zeros(n, dtype=int)

        current_front = np.where(num_dominance == 0)[0].tolist()
        fronts = [current_front]
        local_rank[current_front] = 0

        aux_rank = 1
        while len(current_front) > 0:
            # Vectorized "peeling" of the current front.
            dominated_rows = dominance_matrix[current_front]  # (|front|, n)
            removed = dominated_rows.sum(axis=0)
            num_dominance = num_dominance - removed

            # q joins the next front when its LAST dominator has been processed.
            released = np.flatnonzero((removed > 0) & (num_dominance == 0))

            # Orders the released agents as the original loop-based implementation
            # would (by their last dominator, then by index), keeping results identical
            if released.size > 1:
                last_dominator = (
                    dominated_rows.shape[0]
                    - 1
                    - np.argmax(dominated_rows[::-1][:, released], axis=0)
                )
                released = released[np.lexsort((released, last_dominator))]

            next_front = released.tolist()
            local_rank[released] = aux_rank

            if len(next_front) > 0:
                fronts.append(next_front)
            current_front = next_front
            aux_rank += 1

        return fronts, local_rank

    def _calculate_crowding_distance(
        self, front: List, agents: List[Agent]
    ) -> np.ndarray:
        """Calculates the crowding distance of the agents of a single front.

        Args:
            front: Indices (in `agents`) of the agents that belong to the front.
            agents: List of agents.

        Returns:
            (np.ndarray): Crowding distance of each agent of the front, in the same
                order as `front`. Boundary agents receive an infinite distance.

        """

        n_agents_in_front = len(front)
        distances = np.zeros(n_agents_in_front)

        # Every agent of a front with at most two agents is a boundary one
        if n_agents_in_front <= 2:
            distances[:] = np.inf
            return distances

        n_objectives = len(agents[0].fit)
        fit_values = np.array(
            [agents[idx].fit for idx in front]
        )  # Shape: (len(front), M)

        for m in range(n_objectives):
            obj_m = fit_values[:, m]
            sorted_order = np.argsort(obj_m)
            sorted_obj = obj_m[sorted_order]

            distances[sorted_order[0]] = np.inf
            distances[sorted_order[-1]] = np.inf

            # All agents share the same value: this objective adds no spread
            amplitude = sorted_obj[-1] - sorted_obj[0]
            if amplitude == 0:
                continue

            # Normalized gap between the two neighbours of each interior agent
            distances[sorted_order[1:-1]] += (
                sorted_obj[2:] - sorted_obj[:-2]
            ) / amplitude

        return distances

    def _tournament_selection(
        self, agents: List[Agent], n_selected: int = None
    ) -> np.ndarray:
        """Selects parents through binary tournaments (crowded-comparison operator).

        The lower rank wins and, on ties, the larger crowding distance.

        Args:
            agents: List of agents taking part in the tournaments.
            n_selected: Number of parents to select (defaults to `len(agents)`).

        Returns:
            (np.ndarray): Indices of the selected parents.

        """

        N = len(agents)
        n_selected = N if n_selected is None else n_selected

        competitors_A = np.random.randint(0, N, size=n_selected)
        competitors_B = np.random.randint(0, N, size=n_selected)

        cond_ranking = self.rank[competitors_A] < self.rank[competitors_B]
        cond_tie = (self.rank[competitors_A] == self.rank[competitors_B]) & (
            self.crowding_distance[competitors_A]
            > self.crowding_distance[competitors_B]
        )

        victory_A = cond_ranking | cond_tie
        selected = np.where(victory_A, competitors_A, competitors_B)

        return selected

    def _create_offspring(self, space: _MultiObjectiveSpace) -> List[Agent]:
        """Creates the offspring population through selection, crossover and mutation.

        Args:
            space: A MultiObjectiveSpace object.

        Returns:
            (List[Agent]): `n_agents` new (not yet evaluated) agents.

        """

        N = len(space.agents)

        # Each pair of parents yields two children, so ceil(N / 2) pairs are
        # enough to build an offspring population of exactly N agents.
        n_pairs = (N + 1) // 2
        parents_idx = self._tournament_selection(space.agents, 2 * n_pairs)

        parents1 = [space.agents[int(i)] for i in parents_idx[:n_pairs]]
        parents2 = [space.agents[int(i)] for i in parents_idx[n_pairs:]]

        crossed = self.crossover_operator(parents1, parents2)[:N]
        mutated = self.mutation_operator(crossed)

        return mutated

    def _select_survivors(
        self, combined_population: List[Agent], space: _MultiObjectiveSpace
    ):
        """Selects the next population from the union of parents and offspring.

        Whole fronts are kept while they fit; the front that overflows is truncated
        by keeping its agents with the largest crowding distance.

        Args:
            combined_population: Parents followed by the evaluated offspring.
            space: A MultiObjectiveSpace object.

        Returns:
            (Tuple[List[Agent], np.ndarray, np.ndarray]): Survivors, their ranks and
                their crowding distances.

        """

        fronts, _ = self._fast_non_dominated_sort(combined_population)

        new_population = []
        new_ranks = []
        new_crowding = []
        n_agents = len(space.agents)

        for front_idx, front in enumerate(fronts):
            if len(new_population) >= n_agents:
                break

            crowding = self._calculate_crowding_distance(front, combined_population)

            if len(new_population) + len(front) <= n_agents:
                for idx, cd in zip(front, crowding):
                    new_population.append(combined_population[idx])
                    new_ranks.append(front_idx)
                    new_crowding.append(cd)
            else:
                # Last front only partially fits: keeps its least crowded agents
                remaining = n_agents - len(new_population)

                top_indices = np.argsort(crowding)[-remaining:][::-1]

                for local_idx in top_indices:
                    global_idx = front[local_idx]
                    new_population.append(combined_population[global_idx])
                    new_ranks.append(front_idx)
                    new_crowding.append(crowding[local_idx])

        return (
            new_population,
            np.array(new_ranks, dtype=int),
            np.array(new_crowding, dtype=float),
        )

    def update(self, space: _MultiObjectiveSpace, function) -> None:
        """Performs one generation: creates, evaluates and selects the offspring.

        Args:
            space: A MultiObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        offspring = self._create_offspring(space)
        agent_shape = space.agents[0].position.shape

        for i in range(len(offspring)):
            offspring[i].position = offspring[i].position.reshape(agent_shape)
            offspring[i].fit = function(offspring[i].position).flatten()

        combined_population = space.agents + offspring
        new_pop, new_ranks, new_crowding = self._select_survivors(
            combined_population, space
        )

        space.agents = new_pop
        self.rank = new_ranks
        self.crowding_distance = new_crowding

    def evaluate(self, space, function):
        """Evaluates the initial population and computes its ranks and crowding.

        Offspring are evaluated inside `update`, so this method replaces itself
        with a no-op after its first call.

        Args:
            space: A MultiObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        for agent in space.agents:
            agent.fit = function(agent.position).flatten()

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

        self.evaluate = lambda: None


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
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Tensorized crossover operator applied to the parents.
            mutation_operator: Tensorized mutation operator applied to the children.

        """

        logger.info("Overriding class: MultiObjectiveOptimizer -> NSGA2 (Tensorized).")

        super().__init__()

        self.crossover_operator = crossover_operator or SBXCrossoverTensor(
            env=Environment("numpy", "float32")
        )
        self.mutation_operator = mutation_operator or PolynomialMutationTensor(
            env=Environment("numpy", "float32")
        )
        # Populated in `compile`.
        self._xp = None  # concrete array module (numpy / cupy), NOT the proxy
        self._use_dom_kernel = False  # fused CUDA dominance kernel available?
        self._peel_batch = 1  # fronts peeled between two emptiness checks

        self.build(params)

        self.DTYPE = None
        logger.info("Class overrided.")

    @property
    def rank(self):
        """Non-dominated front index of each agent (0 is the best front)."""
        return self._rank

    @rank.setter
    def rank(self, rank) -> None:
        self._rank = rank

    @property
    def crowding_distance(self):
        """Crowding distance of each agent inside its own front."""
        return self._crowding_distance

    @crowding_distance.setter
    def crowding_distance(self, crowding_distance) -> None:
        self._crowding_distance = crowding_distance

    def compile(self, space: _MultiObjectiveTensorSpace) -> None:
        """Allocates the population buffers and binds the optimizer to the backend.

        `P_tensor` / `F_tensor` hold `2 * n_agents` rows: the current population in
        the first half and the offspring of the generation in the second one.

        Args:
            space: A tensorized MultiObjectiveSpace object.

        """

        xp = space.env.xp
        self.DTYPE = space.env.dtype
        self.xp_module = xp

        self._xp = xp.__wrapped__
        # The operators must run on the same backend as the space.
        self.crossover_operator.env = space.env
        self.mutation_operator.env = space.env
        self._use_dom_kernel = bool(space.env.use_cuda)
        # The emptiness check of the peeling loop only costs a device sync on CUDA.
        self._peel_batch = _PEEL_BATCH_CUDA if space.env.use_cuda else 1

        n = space.n_agents
        self.n_agents = n

        self.n_variables = space.n_variables
        self.n_dimensions = space.n_dimensions
        self.m = space.n_objectives

        self.rank = xp.zeros(n, dtype=xp.int32)
        self.crowding_distance = xp.zeros(n, dtype=self.DTYPE)

        self.P_tensor = xp.zeros(
            (2 * n, self.n_variables, self.n_dimensions), dtype=self.DTYPE
        )
        self.F_tensor = xp.zeros((2 * n, self.m), dtype=self.DTYPE)

        self.P_tensor[:n] = xp.asarray(space.X, dtype=self.DTYPE)
        if space.F is not None:
            self.F_tensor[:n] = xp.asarray(space.F, dtype=self.DTYPE)

        self._LB = space.lb.reshape(1, self.n_variables, self.n_dimensions)
        self._UB = space.ub.reshape(1, self.n_variables, self.n_dimensions)

    def _dominance_matrix(self, F: Any, xp) -> Any:
        """Returns D with D[i, j] = True iff solution i dominates solution j."""
        N, M = F.shape

        if self._use_dom_kernel:
            try:
                idx = xp.arange(N, dtype=xp.int32)
                return _get_dominance_kernel(xp)(
                    idx[:, xp.newaxis],
                    idx[xp.newaxis, :],
                    xp.ascontiguousarray(F),
                    M,
                )
            except Exception as exc:  # pragma: no cover - depends on CUDA runtime
                logger.warning(
                    "NSGA2Tensor: fused dominance kernel unavailable (%s). "
                    "Falling back to blocked array operations.",
                    exc,
                )
                self._use_dom_kernel = False

        # Blocked fallback: rows are processed in chunks so that the (chunk, N, M)
        # temporaries stay below `_DOMINANCE_BLOCK_ELEMENTS` booleans
        D = xp.empty((N, N), dtype=bool)
        chunk = max(1, _DOMINANCE_BLOCK_ELEMENTS // max(1, N * M))
        Fj = F[xp.newaxis, :, :]
        for start in range(0, N, chunk):
            Fi = F[start : start + chunk, xp.newaxis, :]
            D[start : start + chunk] = xp.all(Fi <= Fj, axis=2) & xp.any(
                Fi < Fj, axis=2
            )

        return D

    def _nondominated_sort(self, F: Any, xp) -> Any:
        """Tensorized non-dominated sorting.

        Fronts are peeled one at a time: a single matrix-vector product removes the
        current front from the domination counters of all remaining solutions.

        Args:
            F: Objective values of shape (N, M).
            xp: Array module (numpy / cupy).

        Returns:
            (Any): Front index (rank) of each solution, of shape (N,).

        """

        N = F.shape[0]

        D = self._dominance_matrix(F, xp)
        # c[j]: number of not-yet-ranked solutions that dominate j
        c = D.sum(axis=0).astype(xp.int32)

        Df = D.astype(F.dtype)

        r = xp.zeros(N, dtype=xp.int32)
        k = 0
        p = c == 0  # mask of the current front

        while True:
            # Several fronts are peeled between two emptiness checks, since each
            # check is a blocking device -> host copy on CUDA. Extra iterations on
            # an empty front are harmless: `p` stays all-False.
            for _ in range(self._peel_batch):
                r = xp.where(p, xp.int32(k), r)

                p_float = p.astype(F.dtype)
                dominated_by_front = (p_float @ Df).astype(xp.int32)
                # Subtracting `p` drives the ranked solutions to -1, so they never
                # satisfy `c == 0` again
                c = c - dominated_by_front - p.astype(xp.int32)

                k += 1
                p = c == 0

            if not bool(xp.any(p)):
                break

        return r

    def _crowding_by_rank(self, F: Any, rank: Any, xp) -> Any:
        """Crowding distance of every solution within its own front, in one shot.

        For each objective, the solutions are sorted by (rank, value), so every front
        becomes a contiguous segment and the neighbours of a solution are the
        adjacent positions of that segment.

        Args:
            F: Objective values of shape (N, M).
            rank: Front index of each solution, of shape (N,).
            xp: Array module (numpy / cupy).

        Returns:
            (Any): Crowding distance of shape (N,); boundary solutions get inf.

        """

        N, M = F.shape
        pos = xp.arange(N)
        prev_pos = xp.maximum(pos - 1, 0)
        next_pos = xp.minimum(pos + 1, N - 1)
        rank_key = rank.astype(F.dtype)

        dist = xp.zeros((N, M), dtype=F.dtype)
        for m in range(M):
            f = F[:, m]
            order = xp.lexsort(xp.stack([f, rank_key]))  # last key is the primary one
            r_sorted = rank[order]
            f_sorted = f[order]

            # First / last sorted position of the front each position belongs to
            first = xp.searchsorted(r_sorted, r_sorted, side="left")
            last = xp.searchsorted(r_sorted, r_sorted, side="right") - 1

            value_range = f_sorted[last] - f_sorted[first]
            value_range = xp.where(value_range == 0, 1e-9, value_range)

            # Neighbours may cross a front border only at boundary positions, which
            # are overwritten with inf right below
            gap = (f_sorted[next_pos] - f_sorted[prev_pos]) / value_range
            boundary = (pos == first) | (pos == last)
            gap = xp.where(boundary, xp.inf, gap)

            dist[order, m] = gap

        return xp.sum(dist, axis=1)

    def _combined_rank_and_crowding(self, F: Any, xp):
        """Computes the rank and the crowding distance of every solution.

        Args:
            F: Objective values of shape (N, M).
            xp: Array module (numpy / cupy).

        Returns:
            (Tuple[Any, Any]): Ranks and crowding distances, both of shape (N,).

        """

        rank = self._nondominated_sort(F, xp)
        crowding = self._crowding_by_rank(F, rank, xp)

        return rank, crowding

    def _tournament_selection_indices(self, n: int, xp) -> Any:
        """Selects `n` parents through binary tournaments (crowded comparison).

        Args:
            n: Population size (and number of tournaments).
            xp: Array module (numpy / cupy).

        Returns:
            (Any): Indices of the selected parents, of shape (n,).

        """

        # The random offset in [1, n) guarantees two distinct competitors
        i = xp.random.randint(0, n, size=n)
        j = (i + xp.random.randint(1, n, size=n)) % n

        ri, rj = self.rank[i], self.rank[j]
        ci, cj = self.crowding_distance[i], self.crowding_distance[j]

        prefer_i = (ri < rj) | ((ri == rj) & (ci > cj))
        return xp.where(prefer_i, i, j)

    def _environmental_selection(self, F_combined: Any, xp, n: int):
        """Selects the `n` survivors of the union of parents and offspring.

        Args:
            F_combined: Objective values of the union, of shape (N, M).
            xp: Array module (numpy / cupy).
            n: Number of survivors.

        Returns:
            (Tuple[Any, Any, Any]): Indices of the survivors, and the ranks and
                crowding distances of the whole union.

        """

        rank, crowding = self._combined_rank_and_crowding(F_combined, xp)

        # Sorts by rank and, inside a front, by decreasing crowding distance
        # (`lexsort` uses the last key as the primary one)
        keys = xp.vstack((-crowding, rank))
        order = xp.lexsort(keys)
        best_idx = order[:n]

        return best_idx, rank, crowding

    def update(self, space: _MultiObjectiveTensorSpace, function) -> None:
        """Performs one generation: creates, evaluates and selects the offspring.

        Args:
            space: A tensorized MultiObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        xp = self._xp
        n = self.n_agents

        # Each pair of parents yields two children, so n // 2 pairs are enough
        sel_idx = self._tournament_selection_indices(n, xp)
        half = n // 2

        idx1 = sel_idx[:half]
        idx2 = sel_idx[half : half * 2]

        P1 = self.P_tensor[idx1]
        P2 = self.P_tensor[idx2]

        LB = self._LB[0]
        UB = self._UB[0]

        P_cross = self.crossover_operator(P1, P2, LB, UB)
        P_cross = xp.vstack(P_cross)

        P_children = self.mutation_operator(P_cross, LB, UB)

        # Offspring are written to the second half of the buffers, right after the
        # current population, so that the union needs no concatenation
        num_children = P_children.shape[0]
        self.P_tensor[n : n + num_children] = P_children

        F_children = function(P_children, xp=xp)
        self.F_tensor[n : n + num_children] = F_children

        F_combined = self.F_tensor[: n + num_children]

        best_idx, rank_combined, crowding_combined = self._environmental_selection(
            F_combined, xp, n
        )

        self.P_tensor[:n] = self.P_tensor[best_idx]
        self.F_tensor[:n] = self.F_tensor[best_idx]

        space.X[:n] = self.P_tensor[:n]
        space.F = self.F_tensor[:n]

        self.rank = rank_combined[best_idx]
        self.crowding_distance = crowding_combined[best_idx]

    def evaluate(self, space: _MultiObjectiveTensorSpace, function) -> None:
        """Evaluates the initial population and computes its ranks and crowding.

        Offspring are evaluated inside `update`, so this method replaces itself
        with a no-op after its first call.

        Args:
            space: A tensorized MultiObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        xp = self._xp

        F = function(self.P_tensor[: self.n_agents], xp=xp)
        self.F_tensor[: self.n_agents] = F
        space.F = self.F_tensor[: self.n_agents]

        self.rank, self.crowding_distance = self._combined_rank_and_crowding(
            self.F_tensor[: self.n_agents], xp
        )

        self.evaluate = lambda: None
