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
    """NSGA2 class, inherited from MultiObjectiveOptimizer."""

    def __init__(
        self,
        params: Dict = None,
        crossover_operator=None,
        mutation_operator=None,
    ) -> None:
        super().__init__()
        self.crossover_operator = crossover_operator or SBXCrossover()
        self.mutation_operator = mutation_operator or PolynomialMutation()
        self.build(params)

    @property
    def rank(self) -> np.ndarray:
        return self._rank

    @rank.setter
    def rank(self, rank: np.ndarray) -> None:
        self._rank = np.asarray(rank)

    @property
    def crowding_distance(self) -> np.ndarray:
        return self._crowding_distance

    @crowding_distance.setter
    def crowding_distance(self, crowding_distance: np.ndarray) -> None:
        self._crowding_distance = np.asarray(crowding_distance)

    def compile(self, space: _MultiObjectiveSpace) -> None:
        self.crowding_distance = np.zeros(space.n_agents)

    def _fast_non_dominated_sort(self, agents: List[Agent]):
        n = len(agents)
        population_fitness = np.array([ag.fit for ag in agents])

        P = population_fitness[:, np.newaxis, :]  # (N, 1, M)
        Q = population_fitness[np.newaxis, :, :]  # (1, N, M)

        cond_1 = np.all(P <= Q, axis=2)
        cond_2 = np.any(P < Q, axis=2)
        dominance_matrix = cond_1 & cond_2

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
        n_agents_in_front = len(front)
        distances = np.zeros(n_agents_in_front)

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

            amplitude = sorted_obj[-1] - sorted_obj[0]
            if amplitude == 0:
                continue

            distances[sorted_order[1:-1]] += (
                sorted_obj[2:] - sorted_obj[:-2]
            ) / amplitude

        return distances

    def _tournament_selection(self, agents: List[Agent]) -> np.ndarray:
        N = len(agents)

        competitors_A = np.random.randint(0, N, size=2 * N)
        competitors_B = np.random.randint(0, N, size=2 * N)

        cond_ranking = self.rank[competitors_A] < self.rank[competitors_B]
        cond_tie = (self.rank[competitors_A] == self.rank[competitors_B]) & (
            self.crowding_distance[competitors_A]
            > self.crowding_distance[competitors_B]
        )

        victory_A = cond_ranking | cond_tie
        selected = np.where(victory_A, competitors_A, competitors_B)

        return selected

    def _create_offspring(self, space: _MultiObjectiveSpace) -> List[Agent]:
        parents_idx = self._tournament_selection(space.agents)
        N = len(space.agents)

        parents1 = [space.agents[int(i)] for i in parents_idx[:N]]
        parents2 = [space.agents[int(i)] for i in parents_idx[N : 2 * N]]

        crossed = self.crossover_operator(parents1, parents2)
        mutated = self.mutation_operator(crossed)

        return mutated

    def _select_survivors(
        self, combined_population: List[Agent], space: _MultiObjectiveSpace
    ):
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

        self._xp = xp.__wrapped__
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
        N = F.shape[0]

        D = self._dominance_matrix(F, xp)
        c = D.sum(axis=0).astype(xp.int32)

        Df = D.astype(F.dtype)

        r = xp.zeros(N, dtype=xp.int32)
        k = 0
        p = c == 0

        while True:
            for _ in range(self._peel_batch):
                r = xp.where(p, xp.int32(k), r)

                p_float = p.astype(F.dtype)
                dominated_by_front = (p_float @ Df).astype(xp.int32)
                c = c - dominated_by_front - p.astype(xp.int32)

                k += 1
                p = c == 0

            if not bool(xp.any(p)):
                break

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
            interior_dists = (
                sorted_costs[2:num_valid_elem] - sorted_costs[: num_valid_elem - 2]
            ) / distance_range
            sorted_distances[1 : num_valid_elem - 1, :] = interior_dists

        sorted_distances[0, :] = xp.inf
        if num_valid_elem > 1:
            sorted_distances[num_valid_elem - 1, :] = xp.inf

        distance = xp.zeros_like(costs)
        xp.put_along_axis(distance, rank, sorted_distances, axis=0)

        crowding_distances = xp.where(mask[:, None], distance, -xp.inf)
        crowding_distances = xp.sum(crowding_distances, axis=1)

        return crowding_distances

    def _crowding_by_rank(self, F: Any, rank: Any, xp) -> Any:
        """Crowding distance of every solution within its own front, in one shot."""
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

            first = xp.searchsorted(r_sorted, r_sorted, side="left")
            last = xp.searchsorted(r_sorted, r_sorted, side="right") - 1

            value_range = f_sorted[last] - f_sorted[first]
            value_range = xp.where(value_range == 0, 1e-9, value_range)

            gap = (f_sorted[next_pos] - f_sorted[prev_pos]) / value_range
            boundary = (pos == first) | (pos == last)
            gap = xp.where(boundary, xp.inf, gap)

            dist[order, m] = gap

        return xp.sum(dist, axis=1)

    def _combined_rank_and_crowding(self, F: Any, xp):
        rank = self._nondominated_sort(F, xp)
        crowding = self._crowding_by_rank(F, rank, xp)

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
        xp = self._xp
        n = self.n_agents

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
        xp = self._xp

        F = function(self.P_tensor[: self.n_agents], xp=xp)
        self.F_tensor[: self.n_agents] = F
        space.F = self.F_tensor[: self.n_agents]

        self.rank, self.crowding_distance = self._combined_rank_and_crowding(
            self.F_tensor[: self.n_agents], xp
        )

        self.evaluate = lambda: None
