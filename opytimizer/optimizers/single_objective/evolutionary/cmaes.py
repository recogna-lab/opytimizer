"""Covariance Matrix Adaptation Evolution Strategy-based algorithms.
"""

import copy
import math
import time
from typing import Any, Dict, Optional

import numpy as np


import opytimizer.utils.exception as e
from opytimizer.core import Optimizer, TensorizedOptimizer
from opytimizer.core.function import Function
from opytimizer.core.space import _SingleObjectiveSpace, _SingleObjectiveTensorSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class CMAES(Optimizer):
    """A CMAES class, inherited from Optimizer.

    This is the designed class to define CMA-ES-related
    variables and methods.

    References:
        N. Hansen. The CMA evolution strategy: a comparing review. 
        Towards a new evolutionary computation (2006).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> CMAES.")

        super().__init__()

        self.sigma = 0.5

        self.build(params)

        logger.info("Class overrided.")

    @property
    def sigma(self) -> float:
        """Overall standard deviation (step size)."""

        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if not isinstance(sigma, (float, int)):
            raise e.TypeError("`sigma` should be a float or integer")
        if sigma <= 0:
            raise e.ValueError("`sigma` should be > 0")

        self._sigma = sigma

    def compile(self, space: _SingleObjectiveSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.dim = space.n_variables * space.n_dimensions
        self.pop_size = space.n_agents
        self.mu = self.pop_size // 2

        # Recombination weights
        weights = np.log((self.pop_size + 1) / 2) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mu_eff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        
        self.chi_n = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        # Step-size control parameters
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1, 0) + self.c_sigma
        
        # Covariance matrix adaptation parameters
        self.c_c = (self.mu_eff + 2) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff))
        
        self.decomp_per_iter = max(math.floor(1 / (self.c_1 + self.c_mu) / self.dim / 10), 1)

        # Evolution paths and matrix initialization
        self.iteration = 0
        self.C = np.eye(self.dim)
        self.C_invsqrt = np.eye(self.dim)
        self.B_ret = np.eye(self.dim)
        self.D_ret = np.eye(self.dim)

        self._transform = self.D_ret @ self.B_ret
        
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        
        # Mean initialized as the center of the generated population
        all_positions = np.array([agent.position.flatten() for agent in space.agents])
        self.mean = np.mean(all_positions, axis=0)

    def evaluate(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        for agent in space.agents:
            agent.fit = function(agent.position).squeeze()
           
            
            
            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: _SingleObjectiveSpace) -> None:
        """Wraps CMA-ES over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        self.iteration += 1

        agents_sorted = sorted(space.agents, key=lambda x: x.fit)
        pop_selected = np.array([a.position.flatten() for a in agents_sorted[:self.mu]])
        
        # Updates mean
        new_mean = self.mean + self.weights @ (pop_selected - self.mean)
        delta_mean = new_mean - self.mean
        
        # Updates evolution paths
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (self.C_invsqrt @ delta_mean) / self.sigma
        
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        h_sigma = 1.0 if (norm_p_sigma / math.sqrt(
            1 - (1 - self.c_sigma) ** (2 * self.iteration)) < (1.4 + 2 / (self.dim + 1)) * self.chi_n) else 0.0
        
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff) * delta_mean / self.sigma
        
        # Updates covariance matrix
        y = (pop_selected - self.mean) / self.sigma
        self.C = ((1 - self.c_1 - self.c_mu) * self.C 
                  + self.c_1 * (np.outer(self.p_c, self.p_c) + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C) 
                  + self.c_mu * (y.T * self.weights) @ y)
        
        # Updates step-size
        self.sigma = self.sigma * math.exp(self.c_sigma / self.d_sigma * (norm_p_sigma / self.chi_n - 1))
        self.mean = new_mean

        # Matrix Decomposition
        if self.iteration % self.decomp_per_iter == 0:
            self.C = (self.C + self.C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, a_min=1e-8, a_max=None)
            self.C_invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            self.B_ret = eigvecs.T
            self.D_ret = eigvecs @ np.diag(np.sqrt(eigvals))

            self._transform = self.D_ret @ self.B_ret

        # Generates new population
        Z = np.random.randn(self.pop_size, self.dim)
        new_positions = self.mean + self.sigma * (Z @ self._transform)

        for agent, new_pos in zip(space.agents, new_positions):
            agent.position = new_pos.reshape(agent.position.shape)


class CMAESTensor(Optimizer, TensorizedOptimizer):
    """A CMAESTensor class, inherited from Optimizer and TensorizedOptimizer.

    This is the designed class to define GPU/CPU-agnostic tensorized 
    CMA-ES-related variables and methods.

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> CMAESTensor.")

        super().__init__()

        self.sigma = 0.5

        self.build(params)

        logger.info("Class overrided.")

    @property
    def sigma(self) -> float:
        """Overall standard deviation (step size)."""

        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if not isinstance(sigma, (float, int)):
            raise e.TypeError("`sigma` should be a float or integer")
        if sigma <= 0:
            raise e.ValueError("`sigma` should be > 0")

        self._sigma = sigma

    def compile(self, space: _SingleObjectiveTensorSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A TensorSpace object containing meta-information.

        """

        xp = space.env.xp
        self.dtype = space.env.dtype
        self.dim = space.n_variables * space.n_dimensions
        self.pop_size = space.n_agents
        self.mu = self.pop_size // 2

        weights = xp.log((self.pop_size + 1) / 2) - xp.log(xp.arange(1, self.mu + 1, dtype=self.dtype))
        self.weights = weights / xp.sum(weights)
        self.mu_eff = float(xp.sum(self.weights) ** 2 / xp.sum(self.weights ** 2))
        
        self.chi_n = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1, 0) + self.c_sigma
        self.c_c = (self.mu_eff + 2) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff))
        
        self.decomp_per_iter = max(math.floor(1 / (self.c_1 + self.c_mu) / self.dim / 10), 1)

        self.iteration = 0
        self.C = xp.eye(self.dim, dtype=self.dtype)
        self.C_invsqrt = xp.eye(self.dim, dtype=self.dtype)
        self.B_ret = xp.eye(self.dim, dtype=self.dtype)
        self.D_ret = xp.eye(self.dim, dtype=self.dtype)
        
        self.p_sigma = xp.zeros(self.dim, dtype=self.dtype)
        self.p_c = xp.zeros(self.dim, dtype=self.dtype)
        
        self.mean = xp.mean(space.X.reshape(self.pop_size, self.dim), axis=0)

        self._lb = xp.asarray(space.lb, dtype=self.dtype).reshape(1, -1)
        self._ub = xp.asarray(space.ub, dtype=self.dtype).reshape(1, -1)

        self.global_best_position = xp.zeros((self.dim, 1), dtype=self.dtype)
        self.global_best_fit = xp.asarray(xp.inf, dtype=self.dtype)

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A TensorSpace object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        
        xp = space.env.xp
        space.F = function(space.X.squeeze(), xp)

        best_idx = xp.argmin(space.F)
        best_fit_val = space.F[best_idx]

        improved_global = best_fit_val < self.global_best_fit
        candidate = space.X[best_idx].reshape(-1, 1)
        self.global_best_position = xp.where(improved_global, candidate, self.global_best_position)
        self.global_best_fit = xp.where(improved_global, best_fit_val, self.global_best_fit)
 
        space.best_agent.position = self.global_best_position.reshape(space.best_agent.position.shape)
        space.best_agent.fit = self.global_best_fit

        #space.best_agent.ts = int(time.time())

    def update(self, space: _SingleObjectiveTensorSpace) -> None:
        """Wraps CMA-ES over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        xp = space.env.xp
        self.iteration += 1

        flat_X = space.X.reshape(self.pop_size, self.dim)
        idx_sorted = xp.argsort(space.F)
        pop_selected = flat_X[idx_sorted[:self.mu]]
        
        new_mean = self.mean + self.weights @ (pop_selected - self.mean)
        delta_mean = new_mean - self.mean
        
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + xp.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (self.C_invsqrt @ delta_mean) / self.sigma
        
        norm_p_sigma = xp.linalg.norm(self.p_sigma)

        h_sigma = (
            norm_p_sigma / xp.sqrt(1 - (1 - self.c_sigma) ** (2 * self.iteration))
            < (1.4 + 2 / (self.dim + 1)) * self.chi_n
        ).astype(self.dtype)
        
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * xp.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff) * delta_mean / self.sigma
        
        y = (pop_selected - self.mean) / self.sigma
        self.C = ((1 - self.c_1 - self.c_mu) * self.C 
                  + self.c_1 * (xp.outer(self.p_c, self.p_c) + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C) 
                  + self.c_mu * (y.T * self.weights) @ y)

        self._sigma = self._sigma * xp.exp(self.c_sigma / self.d_sigma * (norm_p_sigma / self.chi_n - 1))
        self.mean = new_mean

        if self.iteration % self.decomp_per_iter == 0:
            self.C = (self.C + self.C.T) / 2
            eigvals, eigvecs = xp.linalg.eigh(self.C)
            eigvals = xp.clip(eigvals, a_min=1e-8, a_max=None)
            self.C_invsqrt = eigvecs @ xp.diag(1.0 / xp.sqrt(eigvals)) @ eigvecs.T
            self.B_ret = eigvecs.T
            self.D_ret = eigvecs @ xp.diag(xp.sqrt(eigvals))

        transform = xp.matmul(self.D_ret, self.B_ret)
        randn_buf = xp.random.randn(self.pop_size, self.dim).astype(self.dtype)
        
        new_X = self.mean + self.sigma * xp.matmul(randn_buf, transform)
        space.X = new_X.reshape(space.X.shape)



_CMAES_GENERATE_KERNEL_F64 = r"""
extern "C" __global__
void generate_population_f64(
    double* __restrict__ X,
    const double* __restrict__ randn_buf,
    const double* __restrict__ mean,
    const double* __restrict__ transform,
    const double* __restrict__ sigma,
    const int dim,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int agent = idx / dim;
    int var = idx % dim;

    double val = 0.0;
    for (int k = 0; k < dim; k++) {
        val += randn_buf[agent * dim + k] * transform[k * dim + var];
    }
    X[idx] = mean[var] + sigma[0] * val;
}
"""

_CMAES_GENERATE_KERNEL_F32 = r"""
extern "C" __global__
void generate_population_f32(
    float* __restrict__ X,
    const float* __restrict__ randn_buf,
    const float* __restrict__ mean,
    const float* __restrict__ transform,
    const float* __restrict__ sigma,
    const int dim,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int agent = idx / dim;
    int var = idx % dim;

    float val = 0.0;
    for (int k = 0; k < dim; k++) {
        val += randn_buf[agent * dim + k] * transform[k * dim + var];
    }
    X[idx] = mean[var] + sigma[0] * val;
}
"""


class CMAESCuda(CMAESTensor):
    """A CMAESCuda class, inherited from CMAESTensor.

    This class offers a highly accelerated GPU-friendly implementation 
    using CuPy RawKernels for CMA-ES Population Generation.

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: CMAESTensor -> CMAESCuda.")

        super().__init__(params, **kwargs)

        self._gen_kernel = None

        logger.info("Class overrided.")

    def compile(self, space: _SingleObjectiveTensorSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A TensorSpace object containing meta-information.

        """

        super().compile(space)

        xp = space.env.xp
        dtype = space.env.dtype

        use_f64 = (dtype == xp.float64 or dtype == 'float64')
        if use_f64:
            self._gen_kernel = xp.RawKernel(_CMAES_GENERATE_KERNEL_F64, 'generate_population_f64')
        else:
            self._gen_kernel = xp.RawKernel(_CMAES_GENERATE_KERNEL_F32, 'generate_population_f32')

        self._kernel_N = self.pop_size * self.dim
        self._block_size = 256
        self._kernel_grid = (self._kernel_N + self._block_size - 1) // self._block_size

    def update(self, space: _SingleObjectiveTensorSpace) -> None:
        """Wraps CMA-ES over all agents and variables using CUDA Raw Kernels.

        Args:
            space: Space containing agents and update-related information.

        """

        xp = space.env.xp
        self.iteration += 1

        flat_X = space.X.reshape(self.pop_size, self.dim)
        idx_sorted = xp.argsort(space.F)
        pop_selected = flat_X[idx_sorted[:self.mu]]
        
        new_mean = self.mean + self.weights @ (pop_selected - self.mean)
        delta_mean = new_mean - self.mean
        
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + xp.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (self.C_invsqrt @ delta_mean) / self.sigma
        
        norm_p_sigma = xp.linalg.norm(self.p_sigma)
        h_sigma = 1.0 if (norm_p_sigma / xp.sqrt(
            1 - (1 - self.c_sigma) ** (2 * self.iteration)) < (1.4 + 2 / (self.dim + 1)) * self.chi_n) else 0.0
        
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * xp.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff) * delta_mean / self.sigma
        
        y = (pop_selected - self.mean) / self.sigma
        self.C = ((1 - self.c_1 - self.c_mu) * self.C 
                  + self.c_1 * (xp.outer(self.p_c, self.p_c) + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C) 
                  + self.c_mu * (y.T * self.weights) @ y)
        
        self._sigma = self._sigma * xp.exp(self.c_sigma / self.d_sigma * (norm_p_sigma / self.chi_n - 1))
        self.mean = new_mean

        if self.iteration % self.decomp_per_iter == 0:
            self.C = (self.C + self.C.T) / 2
            eigvals, eigvecs = xp.linalg.eigh(self.C)
            eigvals = xp.clip(eigvals, a_min=1e-8, a_max=None)
            self.C_invsqrt = eigvecs @ xp.diag(1.0 / xp.sqrt(eigvals)) @ eigvecs.T
            self.B_ret = eigvecs.T
            self.D_ret = eigvecs @ xp.diag(xp.sqrt(eigvals))

        transform = xp.matmul(self.D_ret, self.B_ret)
        randn_buf = xp.random.randn(self.pop_size, self.dim).astype(self.dtype)
        

        # MEM Barrier
        transform_c = xp.ascontiguousarray(transform, dtype=self.dtype)
        randn_buf_c = xp.ascontiguousarray(randn_buf, dtype=self.dtype)
        mean_c = xp.ascontiguousarray(self.mean, dtype=self.dtype)

        sigma_c = xp.ascontiguousarray(xp.asarray(self.sigma, dtype=self.dtype).reshape(1))
        new_X_c = xp.empty((self.pop_size, self.dim), dtype=self.dtype)

        self._gen_kernel(
            (self._kernel_grid,), (self._block_size,),
            (
                new_X_c,
                randn_buf_c,
                mean_c,
                transform_c,
                sigma_c,
                xp.int32(self.dim),
                xp.int32(self._kernel_N)
            )
        )

        space.X = new_X_c.reshape(space.X.shape)