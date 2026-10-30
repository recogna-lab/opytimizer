"""Reference Vector Guided Evolutionary Algorithm"""

from typing import Any, Dict, Optional, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import (
    Environment,
    MultiObjectiveOptimizer,
    TensorizedMultiObjectiveOptimizer,
)
from opytimizer.core.function import Function
from opytimizer.core.space import _MultiObjectiveSpace, _MultiObjectiveTensorSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import (
    PolynomialMutation,
    PolynomialMutationTensor,
    SBXCrossover,
    SBXCrossoverTensor,
)

logger = logging.get_logger(__name__)


class RVEA(MultiObjectiveOptimizer):
    """
    RVEA class, inherited from MultiObjectiveOptimizer.

    References:
        R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization,"
        in IEEE Transactions on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, Oct. 2016, doi: 10.1109/TEVC.2016.2519378.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        crossover_operator=None,
        mutation_operator=None,
        reference_vectors: np.ndarray = None,
        max_generations: int = 250,
        alpha: Union[float, int] = 2.0,
        fr: float = 0.1,
        **kwargs,
    ):
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator. Defaults to SBX with eta = 30
                and crossover probability 1.0, as in Cheng et al. (2016).
            mutation_operator: Mutation operator. Defaults to the polynomial
                mutation with eta = 20 and rate 1 / n_variables.
            reference_vectors: Reference vectors, one per agent.
            max_generations: Number of generations t_max of the run, used by the
                penalty function and the adaptation frequency. It should match
                the stopping criterion.
            alpha: Rate of change of the penalty function.
            fr: Frequency of the reference vector adaptation (fraction of t_max).

        """

        super().__init__()

        logger.info("Overriding class: MultiObjectiveOptimizer -> RVEA (Default).")

        self.crossover_operator = crossover_operator or SBXCrossover(
            eta=30, rate=1.0, n_offspring=2
        )
        # The default mutation rate (1 / n_variables) is only known in `compile`
        self._default_mutation = mutation_operator is None
        self.mutation_operator = mutation_operator or PolynomialMutation(eta=20)
        self.reference_vectors = reference_vectors
        self.current_reference_vectors = reference_vectors.copy()
        self.max_generations = max_generations
        self.currentGeneration = 0
        self.z = None
        self.alpha = alpha
        self.fr = fr

        # Min. angle between each reference vector and its closest neighbour. It
        # only changes when the reference vectors are adapted, so it is cached
        # (see `_update_gammas`)
        self._gammas = None

        self.build(params)

    @property
    def max_generations(self) -> int:
        return self._max_generations

    @max_generations.setter
    def max_generations(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError("`max_generations` should be an integer.")
        if value <= 0:
            raise e.ValueError("`max_generations` should be higher than 0.")

        self._max_generations = value

    @property
    def fr(self) -> float:
        return self._fr

    @fr.setter
    def fr(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError("`fr` should be an integer or a float.")
        if value <= 0:
            raise e.ValueError("`fr` should be higher than 0.")

        self._fr = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Union[float, int]) -> None:
        if not isinstance(value, (float, int)):
            raise e.TypeError("`alpha` should be a float or an integer.")
        if value <= 0:
            raise e.ValueError("`alpha` should be higher than 0.")

        self._alpha = float(value)

    def compile(self, space: _MultiObjectiveSpace):
        if len(self.reference_vectors) != space.n_agents:
            raise e.ValueError(
                "Error: The number of `reference_vectors` should be equal to the number of agents."
            )

        if self._default_mutation:
            self.mutation_operator.rate = 1.0 / space.n_variables

        rv = np.asarray(self.reference_vectors, dtype=float)
        self.reference_vectors = rv / np.linalg.norm(rv, axis=1, keepdims=True)
        self.current_reference_vectors = self.reference_vectors.copy()
        self._update_gammas()

    def _update_gammas(self) -> None:
        """Caches gamma_i, the smallest angle between V_i and any other reference vector."""
        ref_cos = np.clip(
            self.current_reference_vectors @ self.current_reference_vectors.T,
            -1.0,
            1.0,
        )
        ref_angles = np.arccos(ref_cos)
        np.fill_diagonal(ref_angles, np.inf)
        self._gammas = ref_angles.min(axis=1)

    def evaluate(self, space: _MultiObjectiveSpace, function: Function):

        all_fits = []
        for agent in space.agents:
            agent.fit = function(agent.position).squeeze()
            all_fits.append(agent.fit)
        all_fits = np.array(all_fits)
        self.z = np.min(all_fits, axis=0)

        self.evaluate = lambda: None

    def _adapt_reference_vectors(self, space: _MultiObjectiveSpace):
        """Reference vector adaptation strategy — Algorithm 3, eq. (11)."""
        fr_period = max(1, int(self.max_generations * self.fr))
        if self.currentGeneration % fr_period != 0:
            return

        fits = np.array([agent.fit for agent in space.agents])
        z_min = fits.min(axis=0)
        z_max = fits.max(axis=0)

        scale = z_max - z_min
        scale = np.where(scale < 1e-10, 1e-10, scale)

        adapted = self.reference_vectors * scale[None, :]
        norms = np.linalg.norm(adapted, axis=1, keepdims=True)
        self.current_reference_vectors = adapted / norms
        self._update_gammas()

    def update(self, space: _MultiObjectiveSpace, function: Function):
        """Performs one generation: offspring creation, selection and adaptation.

        Args:
            space: A MultiObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        n_target = len(self.reference_vectors)
        n_cur = len(space.agents)

        # N offspring from random pairs, whatever the current population size:
        # every agent is equally likely to be a parent (Section III-B)
        num_pairs = (n_target + 1) // 2
        p1 = np.random.randint(0, n_cur, size=num_pairs)

        if n_cur > 1:
            p2 = (p1 + np.random.randint(1, n_cur, size=num_pairs)) % n_cur
        else:
            p2 = p1.copy()

        parent_indices = np.stack([p1, p2], axis=1)

        current_population = list(space.agents)

        for pair in parent_indices:
            offsprings = self.crossover_operator(
                parent1=space.agents[pair[0]],
                parent2=space.agents[pair[1]],
            )

            if (
                isinstance(offsprings, (list, tuple))
                and len(offsprings) > 0
                and isinstance(offsprings[0], (list, tuple))
            ):
                plain_offsprings = [item for sublist in offsprings for item in sublist]
            else:
                plain_offsprings = (
                    list(offsprings)
                    if isinstance(offsprings, (list, tuple))
                    else [offsprings]
                )

            for i in range(len(plain_offsprings)):
                plain_offsprings[i] = self.mutation_operator(plain_offsprings[i])
                plain_offsprings[i].fit = np.asarray(
                    function(plain_offsprings[i].position)
                ).squeeze()

            current_population.extend(plain_offsprings)

        N_pop = len(current_population)
        M = space.n_objectives

        fits = np.array([a.fit for a in current_population])
        # Ideal point of the combined population P_t U Q_t (eq. 5)
        self.z = fits.min(axis=0)

        translated = fits - self.z
        norms = np.linalg.norm(translated, axis=1)
        safe_norms = np.maximum(norms, 1e-10)

        cos_matrix = (
            translated / safe_norms[:, None]
        ) @ self.current_reference_vectors.T
        cos_matrix = np.clip(cos_matrix, -1.0, 1.0)
        assignments = np.argmax(cos_matrix, axis=1)

        if self._gammas is None:
            self._update_gammas()
        gammas = self._gammas

        assigned_cosines = cos_matrix[np.arange(N_pop), assignments]
        assigned_angles = np.arccos(assigned_cosines)

        t_ratio = self.currentGeneration / self.max_generations
        P = M * (t_ratio**self.alpha) * (assigned_angles / gammas[assignments])

        apd_values = (1.0 + P) * norms

        # One elitist (lowest APD) per non-empty subpopulation (Algorithm 2)
        order = np.lexsort((apd_values, assignments))
        sorted_assignments = assignments[order]
        is_first = np.ones(len(order), dtype=bool)
        is_first[1:] = sorted_assignments[1:] != sorted_assignments[:-1]
        # Empty subpopulations select nobody, so the population may shrink
        space.agents = [current_population[i] for i in order[is_first]]

        self._adapt_reference_vectors(space)

        self.currentGeneration += 1


class RVEATensor(MultiObjectiveOptimizer, TensorizedMultiObjectiveOptimizer):
    """
    Backend-agnostic (NumPy/CuPy) tensorized implementation of RVEA.

    It follows the same algorithm as `RVEA`: only the K survivors of the last
    selection (one per non-empty subpopulation) take part in mating, translation
    and adaptation. Since tensors have a fixed shape, the population arrays keep
    N rows: the first K hold the survivors and the remaining ones repeat them,
    without any device -> host synchronization.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        crossover_operator=None,
        mutation_operator=None,
        reference_vectors: np.ndarray = None,
        max_generations: int = 250,
        alpha: Union[float, int] = 2.0,
        fr: float = 0.1,
    ):
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Tensorized crossover operator. Defaults to SBX with
                eta = 30 and crossover probability 1.0, as in Cheng et al. (2016).
            mutation_operator: Tensorized mutation operator. Defaults to the
                polynomial mutation with eta = 20 and rate 1 / n_variables.
            reference_vectors: Reference vectors, one per agent.
            max_generations: Number of generations t_max of the run, used by the
                penalty function and the adaptation frequency. It should match
                the stopping criterion.
            alpha: Rate of change of the penalty function.
            fr: Frequency of the reference vector adaptation (fraction of t_max).

        """

        super().__init__()
        logger.info("Overriding class: MultiObjectiveOptimizer -> RVEA (Tensor).")

        # The environment of the operators is replaced by the space's in `compile`
        self.crossover_operator = crossover_operator or SBXCrossoverTensor(
            env=Environment("numpy", "float32"), eta=30, rate=1.0
        )
        # The default mutation rate (1 / n_variables) is only known in `compile`
        self._default_mutation = mutation_operator is None
        self.mutation_operator = mutation_operator or PolynomialMutationTensor(
            env=Environment("numpy", "float32"), eta=20
        )

        self.reference_vectors = reference_vectors
        self.current_reference_vectors = reference_vectors.copy()

        self.max_generations = max_generations
        self.currentGeneration = 0
        self.z = None
        self.alpha = alpha
        self.fr = fr

        self._gammas = None
        self._V_unit = None  # unit-norm copy of `current_reference_vectors`
        self._xp = None  # concrete array module (numpy / cupy), NOT the proxy
        self.dtype = None
        self._n_survivors = None  # device scalar K: rows [0, K) are survivors

        self.build(params)

        logger.info("Class overrided.")

    @property
    def max_generations(self) -> int:
        return self._max_generations

    @max_generations.setter
    def max_generations(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError("`max_generations` should be an integer.")
        if value <= 0:
            raise e.ValueError("`max_generations` should be higher than 0.")
        self._max_generations = value

    @property
    def fr(self) -> float:
        return self._fr

    @fr.setter
    def fr(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError("`fr` should be an integer or a float.")
        if value <= 0:
            raise e.ValueError("`fr` should be higher than 0.")
        self._fr = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Union[float, int]) -> None:
        if not isinstance(value, (float, int)):
            raise e.TypeError("`alpha` should be a float or an integer.")
        if value <= 0:
            raise e.ValueError("`alpha` should be higher than 0.")
        self._alpha = float(value)

    def compile(self, space: _MultiObjectiveSpace):
        if len(self.reference_vectors) != space.n_agents:
            raise e.ValueError(
                "The number of `reference_vectors` must equal the number of agents."
            )

        xp = space.env.xp

        self._xp = xp.__wrapped__
        self.dtype = xp.dtype(space.env.dtype)

        # The operators must run on the same backend as the space
        self.crossover_operator.env = space.env
        self.mutation_operator.env = space.env
        if self._default_mutation:
            self.mutation_operator.rate = 1.0 / space.n_variables

        # Every initial agent is a survivor
        self._n_survivors = self._xp.asarray(space.n_agents, dtype=self._xp.int64)

        self.reference_vectors = xp.asarray(self.reference_vectors, dtype=self.dtype)
        self.current_reference_vectors = xp.asarray(
            self.current_reference_vectors, dtype=self.dtype
        )

        self._update_gammas(xp)

    def _update_gammas(self, xp):
        V = self.current_reference_vectors
        nrm = xp.linalg.norm(V, axis=1, keepdims=True)
        V_n = V / xp.maximum(nrm, 1e-10)

        self._V_unit = V_n

        cos = xp.clip(V_n @ V_n.T, -1.0, 1.0)
        ang = xp.arccos(cos)
        xp.fill_diagonal(ang, xp.inf)
        self._gammas = ang.min(axis=1)

    _DENSE_GROUPED_ARGMIN_LIMIT = 1 << 22

    def _grouped_argmin(self, apd_values: Any, assignments: Any, N_ref: int, xp) -> Any:
        """For every reference vector, the index of its lowest-APD member (-1 if empty)."""
        if int(apd_values.shape[0]) * N_ref <= self._DENSE_GROUPED_ARGMIN_LIMIT:
            return self._grouped_argmin_dense(apd_values, assignments, N_ref, xp)

        return self._grouped_argmin_sorted(apd_values, assignments, N_ref, xp)

    def _grouped_argmin_dense(
        self, apd_values: Any, assignments: Any, N_ref: int, xp
    ) -> Any:
        J = xp.arange(N_ref, dtype=xp.int32)
        member = assignments[:, None] == J[None, :]

        apd_mat = xp.where(member, apd_values[:, None], xp.inf)
        col_min = apd_mat.min(axis=0)
        best_raw = apd_mat.argmin(axis=0).astype(xp.int32)

        return xp.where(xp.isfinite(col_min), best_raw, -1)

    def _grouped_argmin_sorted(
        self, apd_values: Any, assignments: Any, N_ref: int, xp
    ) -> Any:
        """Sort by (assignment, apd): every group becomes a contiguous segment with its
        best member first."""
        order = xp.lexsort(xp.stack([apd_values, assignments.astype(apd_values.dtype)]))
        sorted_assignments = assignments[order]

        is_first = xp.concatenate(
            [
                xp.ones(1, dtype=bool),
                sorted_assignments[1:] != sorted_assignments[:-1],
            ]
        )

        target = xp.where(is_first, sorted_assignments, N_ref)
        best = xp.full((N_ref + 1,), -1, dtype=xp.int32)
        best[target] = order.astype(xp.int32)

        return best[:N_ref]

    def _apd_selection(self, X_combined: Any, F_combined: Any, valid: Any, xp):
        """Reference vector guided selection (Algorithm 2) over the valid rows.

        Args:
            X_combined: Decision variables of the parents followed by the offspring.
            F_combined: Objective values of the same rows.
            valid: Mask of the rows that belong to P_t U Q_t (padding rows are False).
            xp: Array module (numpy / cupy).

        Returns:
            (Tuple[Any, Any]): N rows of decision variables and objective values,
                the survivors first, followed by copies of them.

        """

        N_ref = int(self.current_reference_vectors.shape[0])
        M = int(F_combined.shape[1])

        # Ideal point of the combined population P_t U Q_t (eq. 5)
        self.z = xp.where(valid[:, None], F_combined, xp.inf).min(axis=0)

        F_t = F_combined - self.z
        norms = xp.linalg.norm(F_t, axis=1)
        safe_norms = xp.maximum(norms, 1e-10)

        # Cached unit-norm reference vectors (see `_update_gammas`).
        F_unit = F_t / safe_norms[:, None]
        cos_mat = xp.clip(F_unit @ self._V_unit.T, -1.0, 1.0)

        asgn = xp.argmax(cos_mat, axis=1).astype(xp.int32)

        # cos_mat[arange(N_pop), asgn] is just the row maximum; `cos_mat` is
        # already clipped to [-1, 1], so no second clip is needed before arccos.
        asgn_ang = xp.arccos(cos_mat.max(axis=1))

        gamma_i = self._gammas[asgn]

        # M * (t / t_max) ** alpha only depends on host-side counters.
        penalty_scale = M * (
            (self.currentGeneration / self.max_generations) ** self.alpha
        )
        apd = (1.0 + penalty_scale * (asgn_ang / (gamma_i + 1e-10))) * norms
        # Padding rows never win a subpopulation
        apd = xp.where(valid, apd, xp.inf)

        best_idx = self._grouped_argmin(apd, asgn, N_ref, xp)
        safe_idx = xp.where(best_idx >= 0, best_idx, 0)
        # A subpopulation is empty when it has no member or only padding rows
        selected = (best_idx >= 0) & xp.isfinite(apd[safe_idx])

        # Survivors first (one per non-empty subpopulation), then the empty
        # subpopulations, which are filled with copies of the survivors
        order = xp.argsort((~selected).astype(xp.int8), kind="stable")
        survivors = safe_idx[order]
        n_survivors = xp.maximum(selected.sum(), 1)
        rows = survivors[xp.arange(N_ref) % n_survivors]

        self._n_survivors = n_survivors

        return X_combined[rows], F_combined[rows]

    def _adapt_reference_vectors(self, F_agents: Any, xp):
        fr_period = max(1, int(self.max_generations * self.fr))
        if self.currentGeneration % fr_period != 0:
            return

        z_min = F_agents.min(axis=0)
        z_max = F_agents.max(axis=0)
        scale = xp.maximum(z_max - z_min, 1e-10)

        adapted = self.reference_vectors * scale
        norms = xp.linalg.norm(adapted, axis=1, keepdims=True)
        self.current_reference_vectors = adapted / xp.maximum(norms, 1e-10)
        self._update_gammas(xp)

    def evaluate(self, space: _MultiObjectiveTensorSpace, function: Function) -> None:
        xp = self._xp

        space.F = function(space.X, xp=xp)

        self.z = space.F.min(axis=0)

        self.evaluate = lambda: None

    def update(self, space: _MultiObjectiveTensorSpace, function: Function) -> None:
        """Performs one generation: offspring creation, selection and adaptation.

        Args:
            space: A tensorized MultiObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        xp = self._xp
        n = space.X.shape[0]
        half = n // 2

        # Random pairs among the K survivors (rows [0, K)), every survivor being
        # equally likely to be a parent (Section III-B). K stays on the device
        u = xp.random.random(2 * half)
        parents = xp.minimum(
            (u * self._n_survivors).astype(xp.int64), self._n_survivors - 1
        )

        parents1 = space.X[parents[:half]]
        parents2 = space.X[parents[half:]]

        X_off = self.crossover_operator(parents1, parents2, space.lb, space.ub)
        X_off = xp.concatenate(X_off, axis=0)

        X_off = self.mutation_operator(X_off, space.lb, space.ub)

        F_off = function(X_off, xp=xp)
        if not isinstance(F_off, xp.ndarray):
            F_off = xp.asarray(F_off, dtype=self.dtype)

        X_pool = xp.concatenate([space.X, X_off], axis=0)
        F_pool = xp.concatenate([space.F, F_off], axis=0)

        # P_t U Q_t: the survivors and every offspring (padding rows excluded)
        valid = xp.concatenate(
            [xp.arange(n) < self._n_survivors, xp.ones(X_off.shape[0], dtype=bool)]
        )

        X_new, F_new = self._apd_selection(X_pool, F_pool, valid, xp)

        space.X = X_new
        space.F = F_new

        # Padding rows repeat survivors, so they do not change z_min / z_max
        self._adapt_reference_vectors(space.F, xp)
        self.currentGeneration += 1


class RVEACuda(RVEATensor):
    """
    GPU-friendly, fully tensorized implementation of RVEA.

    Identical to `RVEATensor` except for the grouped-argmin step, which can run
    as a single hand-written CUDA kernel.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        crossover_operator=None,
        mutation_operator=None,
        reference_vectors: np.ndarray = None,
        max_generations: int = 250,
        alpha: Union[float, int] = 2.0,
        fr: float = 0.1,
    ):
        # Default operators are created by RVEATensor and bound to the space's
        # environment in `compile`
        super().__init__(
            params,
            crossover_operator,
            mutation_operator,
            reference_vectors,
            max_generations,
            alpha,
            fr,
        )
        logger.info("Overriding class: RVEATensor -> RVEA (CUDA).")

        self._grouped_argmin_kernel = None

    def compile(self, space: _MultiObjectiveSpace):
        super().compile(space)
        self._compile_kernel(self._xp)

    def _compile_kernel(self, xp):
        ctype = "double" if self.dtype == xp.float64 else "float"
        cmax = "1.7976931348623157e+308" if ctype == "double" else "3.402823466e+38f"

        kernel_src = f"""
        extern "C" __global__ void grouped_argmin(
        const {ctype}* __restrict__ apd_values,
        const int* __restrict__ assignments,
              int* __restrict__ best_indices,
        const int N_pop,
        const int N_ref
        ) {{
            int j = (int)blockIdx.x;
            if (j >= N_ref) return;

            extern __shared__ char smem[];
            {ctype}* s_val = ({ctype}*)smem;
            int* s_idx = (int*)(s_val + blockDim.x);

            int tid = (int)threadIdx.x;
            {ctype} local_v  = {cmax};
            int local_i  = -1;

            for (int i = tid; i < N_pop; i += (int)blockDim.x) {{
                if (assignments[i] == j) {{
                    {ctype} v = apd_values[i];
                    if (v < local_v) {{ local_v = v; local_i = i; }}
                }}
            }}
            s_val[tid] = local_v;
            s_idx[tid] = local_i;
            __syncthreads();

            for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {{
                if (tid < s && s_val[tid + s] < s_val[tid]) {{
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }}
                __syncthreads();
            }}

            if (tid == 0) best_indices[j] = s_idx[0];
        }}
        """

        try:
            self._grouped_argmin_kernel = xp.RawKernel(kernel_src, "grouped_argmin")
            logger.info("RVEA [CUDA]: grouped-argmin CUDA kernel compiled OK.")
        except Exception as ex:
            logger.warning(
                f"RVEA [CUDA]: kernel compilation failed ({ex}). "
                "Using vectorised CuPy fallback."
            )
            self._grouped_argmin_kernel = None

    def _grouped_argmin(self, apd_values: Any, assignments: Any, N_ref: int, xp) -> Any:
        if self._grouped_argmin_kernel is None:
            return super()._grouped_argmin(apd_values, assignments, N_ref, xp)

        N_pop = int(apd_values.shape[0])

        best = xp.empty((N_ref,), dtype=xp.int32)
        apd_c = xp.ascontiguousarray(apd_values, dtype=self.dtype)
        asgn_c = xp.ascontiguousarray(assignments, dtype=xp.int32)

        THREADS = 256
        dtype_bytes = 8 if self.dtype == xp.float64 else 4
        SMEM_BYTES = THREADS * (dtype_bytes + 4)

        self._grouped_argmin_kernel(
            (N_ref,),
            (THREADS,),
            (
                apd_c,
                asgn_c,
                best,
                xp.int32(N_pop),
                xp.int32(N_ref),
            ),
            shared_mem=SMEM_BYTES,
        )

        return best
