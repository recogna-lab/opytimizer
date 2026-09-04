"""Reference Vector Guided Evolutionary Algorithm"""

import numpy as np

from typing import Optional, Dict, Any, Union

from opytimizer.core import MultiObjectiveOptimizer, TensorizedMultiObjectiveOptimizer
from opytimizer.core.space import _MultiObjectiveSpace, _MultiObjectiveTensorSpace
from opytimizer.core.function import Function
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation, SBXCrossoverTensor, PolynomialMutationTensor
from opytimizer.core import Environment

import opytimizer.utils.exception as e

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
        **kwargs
    ):
        super().__init__()

        logger.info("Overriding class: MultiObjectiveOptimizer -> RVEA (Default).")

        self.crossover_operator = crossover_operator or SBXCrossover(
            rate=1.0, gene_rate=1.0, n_offspring=2
        )
        self.mutation_operator = mutation_operator or PolynomialMutation(rate=1.0 / 30.0)
        self.reference_vectors = reference_vectors
        self.current_reference_vectors = reference_vectors.copy()
        self.max_generations = max_generations
        self.currentGeneration = 0
        self.z = None
        self.alpha = alpha
        self.fr = fr

        self.build(params)

    @property
    def max_generations(self) -> int:
        return self._max_generations
    @max_generations.setter
    def max_generations(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError('`max_generations` should be an integer.') 
        if value <= 0:
            raise e.ValueError('`max_generations` should be higher than 0.')
        
        self._max_generations = value
        
    @property
    def fr(self) -> float:
        return self._fr
    @fr.setter
    def fr(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError('`fr` should be an integer or a float.') 
        if value <= 0:
            raise e.ValueError('`fr` should be higher than 0.')
        
        self._fr = value
        
    @property
    def alpha(self) -> float:
        return self._alpha
    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isinstance(value, (float)):
            raise e.TypeError('`alpha` should be a float.') 
        if value <= 0:
            raise e.ValueError('`alpha` should be higher than 0.')
        
        self._alpha = value
        

    def compile(self, space: _MultiObjectiveSpace):
        if len(self.reference_vectors) != space.n_agents:
            raise e.ValueError('Error: The number of `reference_vectors` should be equal to the number of agents.')
        
       
    def evaluate(self, space: _MultiObjectiveSpace, function: Function):
       
        all_fits = []
        for agent in space.agents:
            agent.fit = function(agent.position).squeeze()
            all_fits.append(agent.fit)
        all_fits = np.array(all_fits)
        self.z = np.min(all_fits, axis=0)

        self.evaluate = lambda : None 


    def _adapt_reference_vectors(self, space: _MultiObjectiveSpace):
        """Reference vector adaptation strategy — Algorithm 3, eq. (11).

        Adapts current_reference_vectors according to the objective value
        ranges so that solutions stay uniformly distributed even when
        objectives are scaled differently.
        """
        fr_period = max(1, int(self.max_generations * self.fr))
        if self.currentGeneration % fr_period != 0:
            return

        fits = np.array([agent.fit for agent in space.agents])  
        z_min = fits.min(axis=0)                                
        z_max = fits.max(axis=0)                                 

        scale = z_max - z_min
        scale = np.where(scale < 1e-10, 1e-10, scale)           

        # eq. (11):  v_{t+1,i} = (v0_i (.) scale) / ‖v0_i(.) scale‖
        adapted = self.reference_vectors * scale[None, :]        
        norms = np.linalg.norm(adapted, axis=1, keepdims=True) 
        self.current_reference_vectors = adapted / norms

   
    def update(self, space: _MultiObjectiveSpace, function: Function):
        num_pairs = len(space.agents) // 2
        parent_indices = np.random.randint(0, len(space.agents), size=(num_pairs, 2))
        
        current_population = space.agents.copy()

        for pair in parent_indices:
            offsprings = self.crossover_operator(
                parent1=space.agents[pair[0]],
                parent2=space.agents[pair[1]],
            )
            
            if isinstance(offsprings, (list, tuple)) and len(offsprings) > 0 and isinstance(offsprings[0], (list, tuple)):
                plain_offsprings = [item for sublist in offsprings for item in sublist]
            else:
                plain_offsprings = list(offsprings) if isinstance(offsprings, (list, tuple)) else [offsprings]

            for i in range(len(plain_offsprings)):
                plain_offsprings[i] = self.mutation_operator(plain_offsprings[i])         
                plain_offsprings[i].fit = np.asarray(function(plain_offsprings[i].position)).ravel()
                self.z = np.minimum(plain_offsprings[i].fit, self.z)
            current_population.extend(plain_offsprings)

        N_pop = len(current_population)
        N_ref = len(self.current_reference_vectors)
        M  = space.n_objectives

        translated = np.array(
            [agent.fit - self.z for agent in current_population]
        )                                              
        norms = np.linalg.norm(translated, axis=1)     
        safe_norms = np.where(norms < 1e-10, 1e-10, norms)

        cos_matrix  = (translated / safe_norms[:, None]) @ self.current_reference_vectors.T
        cos_matrix  = np.clip(cos_matrix, -1.0, 1.0)       
        assignments = np.argmax(cos_matrix, axis=1)          

        ref_cos  = np.clip(
            self.current_reference_vectors @ self.current_reference_vectors.T,
            -1.0, 1.0,
        )                                                   
        ref_angles = np.arccos(ref_cos)                      
        np.fill_diagonal(ref_angles, np.inf)                 
        gammas = ref_angles.min(axis=1)                    
        
        assigned_cosines = cos_matrix[np.arange(N_pop), assignments]  
        assigned_angles = np.arccos(assigned_cosines)      
        
        t_ratio = self.currentGeneration / self.max_generations
        P = M * (t_ratio ** self.alpha) * (assigned_angles / gammas[assignments])  

        apd_values = (1.0 + P) * norms                       
       
        new_agents = []
        for j in range(N_ref):
            mask = assignments == j
            if not mask.any():
                continue    
            indices  = np.where(mask)[0]
            best_idx = indices[np.argmin(apd_values[indices])]
            new_agents.append(current_population[best_idx])

        space.agents = new_agents
        self._adapt_reference_vectors(space)
        self.currentGeneration += 1


class RVEATensor(MultiObjectiveOptimizer, TensorizedMultiObjectiveOptimizer):
    """
        Backend-agnostic (NumPy/CuPy) tensorized implementation of RVEA, based on:
        Z. Liang, T. Jiang, K. Sun and R. Cheng, "GPU-accelerated Evolutionary
        Multiobjective Optimization Using Tensorized RVEA," in Proceedings of
        the Genetic and Evolutionary Computation Conference (GECCO '24), 2024,
        doi: 10.1145/3638529.3654223.
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
        super().__init__()
        logger.info("Overriding class: MultiObjectiveOptimizer -> RVEA (Tensor).")
 
        
 
        self.crossover_operator = crossover_operator or SBXCrossoverTensor(
             rate=1.0, gene_rate=1.0
        )
        self.mutation_operator = mutation_operator or PolynomialMutationTensor(
            rate=1.0 / 30.0
        )
 
        self.reference_vectors = reference_vectors
        self.current_reference_vectors = reference_vectors.copy()
 
        self.max_generations = max_generations
        self.currentGeneration = 0
        self.z = None
        self.alpha = alpha
        self.fr = fr
 
        self._gammas = None
        self.dtype = None
 
        self.build(params)
 
        logger.info("Class overrided.")
 
    @property
    def max_generations(self) -> int:
        return self._max_generations
 
    @max_generations.setter
    def max_generations(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError('`max_generations` should be an integer.')
        if value <= 0:
            raise e.ValueError('`max_generations` should be higher than 0.')
        self._max_generations = value
 
    @property
    def fr(self) -> float:
        return self._fr
 
    @fr.setter
    def fr(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError('`fr` should be an integer or a float.')
        if value <= 0:
            raise e.ValueError('`fr` should be higher than 0.')
        self._fr = value
 
    @property
    def alpha(self) -> float:
        return self._alpha
 
    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('`alpha` should be a float.')
        if value <= 0:
            raise e.ValueError('`alpha` should be higher than 0.')
        self._alpha = value
 
    def compile(self, space: _MultiObjectiveSpace):
        if len(self.reference_vectors) != space.n_agents:
            raise e.ValueError(
                'The number of `reference_vectors` must equal the number of agents.'
            )
 
        xp = space.env.xp
        self.dtype = xp.float32
 
        self.reference_vectors = xp.asarray(
            self.reference_vectors, dtype=xp.float32
        )
        self.current_reference_vectors = xp.asarray(
            self.current_reference_vectors, dtype=xp.float32
        )
 
        self._update_gammas(xp)
 
    def _update_gammas(self, xp):
        V = self.current_reference_vectors
        nrm = xp.linalg.norm(V, axis=1, keepdims=True)
        V_n = V / xp.where(nrm < xp.asarray(1e-10, dtype=self.dtype),
                             xp.asarray(1e-10, dtype=self.dtype), nrm)
 
        cos = xp.clip(V_n @ V_n.T, xp.asarray(-1.0, dtype=self.dtype), xp.asarray(1.0, dtype=self.dtype))
        ang = xp.arccos(cos)
        xp.fill_diagonal(ang, xp.asarray(xp.inf, dtype=self.dtype))
        self._gammas = ang.min(axis=1)
 
    def _grouped_argmin(self, apd_values: Any, assignments: Any, N_ref: int, xp) -> Any:
        """
        Pure-tensor grouped-argmin: for each of the N_ref reference-vector
        groups, finds the index (into the pooled population) of the member
        with the smallest APD value, or -1 if the group is empty.
        """
        J = xp.arange(N_ref, dtype=xp.int32)
        member = assignments[:, None] == J[None, :]
 
        apd_mat = xp.where(member, apd_values[:, None],
                           xp.asarray(float('inf'), dtype=self.dtype))
        col_min = apd_mat.min(axis=0)
        best_raw = apd_mat.argmin(axis=0).astype(xp.int32)
 
        best = xp.where(xp.isfinite(col_min), best_raw,
                        xp.full((N_ref,), xp.int32(-1), dtype=xp.int32))
 
        return best
 
    def _apd_selection(self, X_combined: Any, F_combined: Any, xp, X_, F_):
        """
        Computes the Angle-Penalized Distance selection (Algorithm 1 of the
        paper) and returns a population of FIXED size `N_ref` (== n_agents).
        """
        N_pop = X_combined.shape[0]
        N_ref = int(self.current_reference_vectors.shape[0])
        M = int(F_combined.shape[1])
 
        F_t = F_combined - self.z
        norms  = xp.linalg.norm(F_t, axis=1)
        safe_norms = xp.where(norms < xp.asarray(1e-10, dtype=self.dtype),
                              xp.asarray(1e-10, dtype=self.dtype), norms)
 
        V_nrm = xp.linalg.norm(self.current_reference_vectors,
                                  axis=1, keepdims=True)
        V_unit = self.current_reference_vectors / xp.where(
            V_nrm < xp.asarray(1e-10, dtype=self.dtype), xp.asarray(1e-10, dtype=self.dtype), V_nrm
        )
 
        F_unit = F_t / safe_norms[:, None]
        cos_mat = xp.clip(F_unit @ V_unit.T,
                          xp.asarray(-1.0, dtype=self.dtype), xp.asarray(1.0, dtype=self.dtype))
 
        asgn = xp.argmax(cos_mat, axis=1).astype(xp.int32)
 
        asgn_cos = cos_mat[xp.arange(N_pop), asgn]
        asgn_ang = xp.arccos(xp.clip(asgn_cos,
                                      xp.asarray(-1.0, dtype=self.dtype),
                                      xp.asarray(1.0, dtype=self.dtype)))
 
        gamma_i = self._gammas[asgn]
 
        t_rat = xp.asarray(self.currentGeneration / self.max_generations, dtype=self.dtype)
        P = (xp.asarray(M, dtype=self.dtype)
             * (t_rat ** xp.asarray(self.alpha, dtype=self.dtype))
             * (asgn_ang / (gamma_i + xp.asarray(1e-10, dtype=self.dtype))))
 
        apd = (xp.asarray(1.0, dtype=self.dtype) + P) * norms
 
        best_idx = self._grouped_argmin(apd, asgn, N_ref, xp)   # shape (N_ref,); -1 == no candidate
        valid = best_idx >= 0
 
        safe_idx = xp.where(valid, best_idx, xp.int32(0))
        X_sel = X_combined[safe_idx]
        F_sel = F_combined[safe_idx]
 
        X_new = xp.where(valid[:, None], X_sel, X_)
        F_new = xp.where(valid[:, None], F_sel, F_)
 
        return X_new, F_new
 
    def _adapt_reference_vectors(self, F_agents: Any, xp):
        fr_period = max(1, int(self.max_generations * self.fr))
        if self.currentGeneration % fr_period != 0:
            return
 
        z_min = F_agents.min(axis=0)
        z_max = F_agents.max(axis=0)
        scale = xp.where(
            (z_max - z_min) < xp.asarray(1e-10, dtype=self.dtype),
            xp.asarray(1e-10, dtype=self.dtype),
            z_max - z_min,
        )
 
        adapted = self.reference_vectors * scale
        norms = xp.linalg.norm(adapted, axis=1, keepdims=True)
        self.current_reference_vectors = adapted / xp.where(
            norms < xp.asarray(1e-10, dtype=self.dtype), xp.asarray(1e-10, dtype=self.dtype), norms
        )
        self._update_gammas(xp)
 
    def evaluate(self, space: _MultiObjectiveTensorSpace, function: Function) -> None:
        xp = space.env.xp
 
        space.F = function(space.X, xp=xp)
 
        self.z = space.F.min(axis=0)
 
        self.evaluate = lambda : None
 
    def update(self, space: _MultiObjectiveTensorSpace, function: Function) -> None:
        xp = space.env.xp
        n = space.X.shape[0]
        half = n // 2
 
        perm = xp.random.permutation(n)
        idx1 = perm[:half]
        idx2 = perm[half:2 * half]
 
        parents1 = space.X[idx1]
        parents2 = space.X[idx2]
 
        X_off = self.crossover_operator(parents1, parents2, space.lb, space.ub)
        X_off = xp.concatenate(X_off, axis=0)
 
        X_off = self.mutation_operator(X_off, space.lb, space.ub)
 
        F_off = function(X_off, xp=xp)
        if not isinstance(F_off, xp.ndarray):
            F_off = xp.asarray(F_off, dtype=self.dtype)
 
        self.z = xp.minimum(self.z, F_off.min(axis=0))
 
        X_pool = xp.concatenate([space.X, X_off], axis=0)
        F_pool = xp.concatenate([space.F, F_off], axis=0)
 
        X_new, F_new = self._apd_selection(X_pool, F_pool, xp, space.X, space.F)
 
        space.X = X_new
        space.F = F_new
 
        self._adapt_reference_vectors(space.F, xp)
        self.currentGeneration += 1




class RVEACuda(MultiObjectiveOptimizer, TensorizedMultiObjectiveOptimizer):
    """
        GPU-friendly, fully tensorized implementation of RVEA, based on:
        Z. Liang, T. Jiang, K. Sun and R. Cheng, "GPU-accelerated Evolutionary
        Multiobjective Optimization Using Tensorized RVEA," in Proceedings of
        the Genetic and Evolutionary Computation Conference (GECCO '24), 2024,
        doi: 10.1145/3638529.3654223.
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
        super().__init__()
        logger.info("Overriding class: MultiObjectiveOptimizer -> RVEA (CUDA).")

        self.crossover_operator = crossover_operator or SBXCrossoverTensor(
            env=Environment('cupy', 'float32'), rate=1.0, gene_rate=1.0
        )
        self.mutation_operator = mutation_operator or PolynomialMutationTensor(rate=1.0 / 30.0, env=Environment('cupy', 'float32'))

        self.reference_vectors = reference_vectors
        self.current_reference_vectors = reference_vectors.copy()

        self.max_generations = max_generations
        self.currentGeneration = 0
        self.z = None
        self.alpha = alpha
        self.fr = fr

        self._gammas = None
        self._grouped_argmin_kernel = None
        self.dtype = None

        self.build(params)

        logger.info("Class overrided.")

    @property
    def max_generations(self) -> int:
        return self._max_generations

    @max_generations.setter
    def max_generations(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError('`max_generations` should be an integer.')
        if value <= 0:
            raise e.ValueError('`max_generations` should be higher than 0.')
        self._max_generations = value

    @property
    def fr(self) -> float:
        return self._fr

    @fr.setter
    def fr(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError('`fr` should be an integer or a float.')
        if value <= 0:
            raise e.ValueError('`fr` should be higher than 0.')
        self._fr = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('`alpha` should be a float.')
        if value <= 0:
            raise e.ValueError('`alpha` should be higher than 0.')
        self._alpha = value

    def compile(self, space: _MultiObjectiveSpace):
        if len(self.reference_vectors) != space.n_agents:
            raise e.ValueError(
                'The number of `reference_vectors` must equal the number of agents.'
            )

        xp = space.env.xp
        self.dtype = xp.float32

        self.reference_vectors = xp.asarray(
            self.reference_vectors, dtype=xp.float32
        )
        self.current_reference_vectors = xp.asarray(
            self.current_reference_vectors, dtype=xp.float32
        )

        self._update_gammas(xp)
        self._compile_kernel(xp)


    def _compile_kernel(self, xp):
        ctype = 'double' if self.dtype == xp.float64 else 'float'
        cmax = '1.7976931348623157e+308' if ctype == 'double' else '3.402823466e+38f'

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
            self._grouped_argmin_kernel = xp.RawKernel(
                kernel_src, 'grouped_argmin'
            )
            logger.info("RVEA [CUDA]: grouped-argmin CUDA kernel compiled OK.")
        except Exception as ex:
            logger.warning(
                f"RVEA [CUDA]: kernel compilation failed ({ex}). "
                "Using vectorised CuPy fallback (higher memory use for large r)."
            )
            self._grouped_argmin_kernel = None

    def _update_gammas(self, xp):
        V = self.current_reference_vectors
        nrm = xp.linalg.norm(V, axis=1, keepdims=True)
        V_n = V / xp.where(nrm < xp.asarray(1e-10, dtype=self.dtype),
                             xp.asarray(1e-10, dtype=self.dtype), nrm)

        cos = xp.clip(V_n @ V_n.T, xp.asarray(-1.0, dtype=self.dtype), xp.asarray(1.0, dtype=self.dtype))
        ang = xp.arccos(cos)
        xp.fill_diagonal(ang, xp.asarray(xp.inf, dtype=self.dtype))
        self._gammas = ang.min(axis=1)


    def _grouped_argmin(self, apd_values: Any, assignments: Any, N_ref: int, xp) -> Any:
        N_pop = int(apd_values.shape[0])
        best  = xp.full((N_ref,), xp.int32(-1), dtype=xp.int32)

        if self._grouped_argmin_kernel is not None:
            THREADS = 256
            dtype_bytes = 8 if self.dtype == xp.float64 else 4
            SMEM_BYTES = THREADS * (dtype_bytes + 4)

            self._grouped_argmin_kernel(
                (N_ref,), (THREADS,),
                (
                    apd_values.astype(self.dtype),
                    assignments.astype(xp.int32),
                    best,
                    xp.int32(N_pop),
                    xp.int32(N_ref),
                ),
                shared_mem=SMEM_BYTES,
            )
        else:
            J = xp.arange(N_ref, dtype=xp.int32)
            member = assignments[:, None] == J[None, :]

            apd_mat = xp.where(member, apd_values[:, None],
                               xp.asarray(float('inf'), dtype=self.dtype))
            col_min = apd_mat.min(axis=0)
            best_raw = apd_mat.argmin(axis=0).astype(xp.int32)

            best = xp.where(xp.isfinite(col_min), best_raw,
                            xp.full((N_ref,), xp.int32(-1), dtype=xp.int32))

        return best

    def _apd_selection(self, X_combined: Any, F_combined: Any, xp, X_, F_):
        """
        Computes the Angle-Penalized Distance selection (Algorithm 1 of the
        paper) and returns a population of FIXED size `N_ref` (== n_agents).
        """
        N_pop = X_combined.shape[0]
        N_ref = int(self.current_reference_vectors.shape[0])
        M = int(F_combined.shape[1])

        F_t = F_combined - self.z
        norms  = xp.linalg.norm(F_t, axis=1)
        safe_norms = xp.where(norms < xp.asarray(1e-10, dtype=self.dtype),
                              xp.asarray(1e-10, dtype=self.dtype), norms)

        V_nrm = xp.linalg.norm(self.current_reference_vectors,
                                  axis=1, keepdims=True)
        V_unit = self.current_reference_vectors / xp.where(
            V_nrm < xp.asarray(1e-10, dtype=self.dtype), xp.asarray(1e-10, dtype=self.dtype), V_nrm
        )

        F_unit = F_t / safe_norms[:, None]
        cos_mat = xp.clip(F_unit @ V_unit.T,
                          xp.asarray(-1.0, dtype=self.dtype), xp.asarray(1.0, dtype=self.dtype))

        asgn = xp.argmax(cos_mat, axis=1).astype(xp.int32)

        asgn_cos = cos_mat[xp.arange(N_pop), asgn]
        asgn_ang = xp.arccos(xp.clip(asgn_cos,
                                      xp.asarray(-1.0, dtype=self.dtype),
                                      xp.asarray(1.0, dtype=self.dtype)))

        gamma_i = self._gammas[asgn]

        t_rat = xp.asarray(self.currentGeneration / self.max_generations, dtype=self.dtype)
        P = (xp.asarray(M, dtype=self.dtype)
             * (t_rat ** xp.asarray(self.alpha, dtype=self.dtype))
             * (asgn_ang / (gamma_i + xp.asarray(1e-10, dtype=self.dtype))))

        apd = (xp.asarray(1.0, dtype=self.dtype) + P) * norms

        best_idx = self._grouped_argmin(apd, asgn, N_ref, xp)   # shape (N_ref,); -1 == no candidate
        valid = best_idx >= 0

        safe_idx = xp.where(valid, best_idx, xp.int32(0))
        X_sel = X_combined[safe_idx]
        F_sel = F_combined[safe_idx]

        X_new = xp.where(valid[:, None], X_sel, X_)
        F_new = xp.where(valid[:, None], F_sel, F_)

        return X_new, F_new

    def _adapt_reference_vectors(self, F_agents: Any, xp):
        fr_period = max(1, int(self.max_generations * self.fr))
        if self.currentGeneration % fr_period != 0:
            return

        z_min = F_agents.min(axis=0)
        z_max = F_agents.max(axis=0)
        scale = xp.where(
            (z_max - z_min) < xp.asarray(1e-10, dtype=self.dtype),
            xp.asarray(1e-10, dtype=self.dtype),
            z_max - z_min,
        )

        adapted = self.reference_vectors * scale
        norms = xp.linalg.norm(adapted, axis=1, keepdims=True)
        self.current_reference_vectors = adapted / xp.where(
            norms < xp.asarray(1e-10, dtype=self.dtype), xp.asarray(1e-10, dtype=self.dtype), norms
        )
        self._update_gammas(xp)

    def evaluate(self, space: _MultiObjectiveTensorSpace, function: Function) -> None:
        xp = space.env.xp
        
        space.F = function(space.X, xp=xp)

        self.z = space.F.min(axis=0)

        self.evaluate = lambda : None

    
    def update(self, space: _MultiObjectiveTensorSpace, function: Function) -> None:
        xp = space.env.xp
        n = space.X.shape[0]
        half = n // 2

        perm = xp.random.permutation(n)
        idx1 = perm[:half]
        idx2 = perm[half:2 * half]

        parents1 = space.X[idx1]
        parents2 = space.X[idx2]

        X_off = self.crossover_operator(parents1, parents2, space.lb, space.ub)
        X_off = xp.concatenate(X_off, axis=0)

        X_off = self.mutation_operator(X_off, space.lb, space.ub)

        F_off = function(X_off, xp=xp)
        if not isinstance(F_off, xp.ndarray):
            F_off = xp.asarray(F_off, dtype=self.dtype)

        self.z = xp.minimum(self.z, F_off.min(axis=0))

        X_pool = xp.concatenate([space.X, X_off], axis=0)
        F_pool = xp.concatenate([space.F, F_off], axis=0)

        X_new, F_new = self._apd_selection(X_pool, F_pool, xp, space.X, space.F)

        space.X = X_new
        space.F = F_new

        self._adapt_reference_vectors(space.F, xp)
        self.currentGeneration += 1
