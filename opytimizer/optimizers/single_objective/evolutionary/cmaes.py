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


def _randn(xp, size, dtype):
    """Standard-normal samples directly in `dtype`.
    """
    if xp.__name__ == "cupy":
        return xp.random.standard_normal(size, dtype=dtype)

    return xp.random.standard_normal(size).astype(dtype)


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
        self.mu_eff = np.sum(self.weights) ** 2 / np.sum(self.weights**2)

        self.chi_n = math.sqrt(self.dim) * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2)
        )

        # Step-size control parameters
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = (
            1
            + 2 * max(math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1, 0)
            + self.c_sigma
        )

        # Covariance matrix adaptation parameters
        self.c_c = (self.mu_eff + 2) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2
            * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((self.dim + 2) ** 2 + self.mu_eff),
        )

        self.decomp_per_iter = max(
            math.floor(1 / (self.c_1 + self.c_mu) / self.dim / 10), 1
        )

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
        pop_selected = np.array(
            [a.position.flatten() for a in agents_sorted[: self.mu]]
        )

        # Updates mean
        new_mean = self.mean + self.weights @ (pop_selected - self.mean)
        delta_mean = new_mean - self.mean

        # Updates evolution paths
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * (self.C_invsqrt @ delta_mean) / self.sigma

        norm_p_sigma = np.linalg.norm(self.p_sigma)
        h_sigma = (
            1.0
            if (
                norm_p_sigma / math.sqrt(1 - (1 - self.c_sigma) ** (2 * self.iteration))
                < (1.4 + 2 / (self.dim + 1)) * self.chi_n
            )
            else 0.0
        )

        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * delta_mean / self.sigma

        # Updates covariance matrix
        y = (pop_selected - self.mean) / self.sigma
        self.C = (
            (1 - self.c_1 - self.c_mu) * self.C
            + self.c_1
            * (
                np.outer(self.p_c, self.p_c)
                + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C
            )
            + self.c_mu * (y.T * self.weights) @ y
        )

        # Updates step-size
        self.sigma = self.sigma * math.exp(
            self.c_sigma / self.d_sigma * (norm_p_sigma / self.chi_n - 1)
        )
        self.mean = new_mean

        # Matrix Decomposition
        if self.iteration % self.decomp_per_iter == 0:
            self.C = (self.C + self.C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, a_min=1e-8, a_max=None)

            sqrt_eig = np.sqrt(eigvals)
            self.C_invsqrt = (eigvecs * (1.0 / sqrt_eig)) @ eigvecs.T
            self.B_ret = eigvecs.T
            self.D_ret = eigvecs * sqrt_eig

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
        self._xp = None  # concrete array module (numpy / cupy), NOT the proxy

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

        weights = xp.log((self.pop_size + 1) / 2) - xp.log(
            xp.arange(1, self.mu + 1, dtype=self.dtype)
        )
        self.weights = weights / xp.sum(weights)
        self.mu_eff = float(xp.sum(self.weights) ** 2 / xp.sum(self.weights**2))

        self.chi_n = math.sqrt(self.dim) * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2)
        )
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = (
            1
            + 2 * max(math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1, 0)
            + self.c_sigma
        )
        self.c_c = (self.mu_eff + 2) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2
            * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((self.dim + 2) ** 2 + self.mu_eff),
        )

        self.decomp_per_iter = max(
            math.floor(1 / (self.c_1 + self.c_mu) / self.dim / 10), 1
        )

        # Concrete module, resolved once. `space.env.xp` is a proxy whose every
        # attribute access goes through `__getattr__`.
        self._xp = xp.__wrapped__

        # Constants that depend only on the hyper-parameters, evaluated once on the
        # host.
        self._sqrt_cs = math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        self._sqrt_cc = math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
        self._cc_term = self.c_c * (2 - self.c_c)
        self._h_threshold = (1.4 + 2 / (self.dim + 1)) * self.chi_n
        self._sigma_rate = self.c_sigma / self.d_sigma
        self._cmu_weights = self.c_mu * self.weights  # c_mu folded into the weights

        self.iteration = 0
        self.C = xp.eye(self.dim, dtype=self.dtype)
        self.C_invsqrt = xp.eye(self.dim, dtype=self.dtype)
        self.B_ret = xp.eye(self.dim, dtype=self.dtype)
        self.D_ret = xp.eye(self.dim, dtype=self.dtype)

        # Cached D @ B (identity until the first decomposition); see `_adapt_distribution`.
        self._transform = xp.eye(self.dim, dtype=self.dtype)

        self.p_sigma = xp.zeros(self.dim, dtype=self.dtype)
        self.p_c = xp.zeros(self.dim, dtype=self.dtype)

        self.mean = xp.mean(space.X.reshape(self.pop_size, self.dim), axis=0)

        self._lb = xp.asarray(space.lb, dtype=self.dtype).reshape(
            1, space.n_variables, space.n_dimensions
        )
        self._ub = xp.asarray(space.ub, dtype=self.dtype).reshape(
            1, space.n_variables, space.n_dimensions
        )

        self.global_best_position = xp.zeros(
            (space.n_variables, space.n_dimensions), dtype=self.dtype
        )
        self.global_best_fit = xp.asarray(xp.inf, dtype=self.dtype)

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A TensorSpace object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        xp = self._xp
        space.F = function(space.X, xp)

        best_idx = xp.argmin(space.F)
        best_fit_val = space.F[best_idx]

        improved_global = best_fit_val < self.global_best_fit
        candidate = space.X[best_idx]
        self.global_best_position = xp.where(
            improved_global, candidate, self.global_best_position
        )
        self.global_best_fit = xp.where(
            improved_global, best_fit_val, self.global_best_fit
        )

        space.best_agent.position = self.global_best_position.reshape(
            space.best_agent.position.shape
        )
        space.best_agent.fit = self.global_best_fit

        space.best_agent.ts = int(time.time())

    def update(self, space: _SingleObjectiveTensorSpace) -> None:
        """Wraps CMA-ES over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        self._adapt_distribution(space)
        self._sample(space)

    def _adapt_distribution(self, space: _SingleObjectiveTensorSpace) -> None:
        """Updates mean, evolution paths, covariance matrix and step size.

        Shared by `CMAESTensor` and `CMAESCuda`.
        """

        xp = self._xp
        self.iteration += 1

        flat_X = space.X.reshape(self.pop_size, self.dim)
        idx_sorted = xp.argsort(space.F)
        pop_selected = flat_X[idx_sorted[: self.mu]]

        # (pop_selected - mean) is needed twice (mean update and rank-mu update).
        y_raw = pop_selected - self.mean
        new_mean = self.mean + self.weights @ y_raw
        delta_mean = new_mean - self.mean

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + self._sqrt_cs * (
            self.C_invsqrt @ delta_mean
        ) / self.sigma

        norm_p_sigma = xp.linalg.norm(self.p_sigma)

        # Heaviside function h_sigma
        h_limit = self._h_threshold * math.sqrt(
            1 - (1 - self.c_sigma) ** (2 * self.iteration)
        )
        h_sigma = (norm_p_sigma < h_limit).astype(self.dtype)

        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * self._sqrt_cc * delta_mean / self.sigma

        # Covariance update
        y = y_raw / self.sigma
        c_coef = (1 - self.c_1 - self.c_mu) + self.c_1 * (1 - h_sigma) * self._cc_term
        C = c_coef * self.C
        C += self.c_1 * xp.outer(self.p_c, self.p_c)
        C += (y.T * self._cmu_weights) @ y
        self.C = C

        self._sigma = self._sigma * xp.exp(
            self._sigma_rate * (norm_p_sigma / self.chi_n - 1)
        )
        self.mean = new_mean

        if self.iteration % self.decomp_per_iter == 0:
            self.C = (self.C + self.C.T) / 2
            eigvals, eigvecs = xp.linalg.eigh(self.C)
            eigvals = xp.clip(eigvals, a_min=1e-8, a_max=None)

            # Column scaling by broadcasting instead of `eigvecs @ xp.diag(s) @
            # eigvecs.T`: same values, without materializing the diagonal matrix
            sqrt_eig = xp.sqrt(eigvals)
            self.C_invsqrt = (eigvecs * (1.0 / sqrt_eig)) @ eigvecs.T
            self.B_ret = eigvecs.T
            self.D_ret = eigvecs * sqrt_eig

            # `D @ B` is a d x d x d product. It only changes when the covariance
            # is decomposed, so it is cached here
            self._transform = xp.matmul(self.D_ret, self.B_ret)

    def _sample(self, space: _SingleObjectiveTensorSpace) -> None:
        """Samples the next population: mean + sigma * (Z @ transform)."""

        xp = self._xp
        randn_buf = _randn(xp, (self.pop_size, self.dim), self.dtype)

        # Scale and shift in place on the GEMM result
        new_X = xp.matmul(randn_buf, self._transform)
        new_X *= self.sigma
        new_X += self.mean

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

    # If True, population sampling uses a cuBLAS GEMM (as `CMAESTensor` does)
    # instead of the Raw Kernel. See `_sample`.
    use_gemm = False

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: CMAESTensor -> CMAESCuda.")

        super().__init__(params, **kwargs)

        self._gen_kernel = None
        self._transform_c = None  # C-contiguous copy of `_transform` for the kernel
        self._transform_src = None  # the `_transform` object `_transform_c` was built from

        logger.info("Class overrided.")

    def compile(self, space: _SingleObjectiveTensorSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A TensorSpace object containing meta-information.

        """

        super().compile(space)

        xp = space.env.xp
        dtype = space.env.dtype

        use_f64 = dtype == xp.float64 or dtype == "float64"
        if use_f64:
            self._gen_kernel = xp.RawKernel(
                _CMAES_GENERATE_KERNEL_F64, "generate_population_f64"
            )
        else:
            self._gen_kernel = xp.RawKernel(
                _CMAES_GENERATE_KERNEL_F32, "generate_population_f32"
            )

        self._kernel_N = self.pop_size * self.dim
        self._block_size = 256
        self._kernel_grid = (self._kernel_N + self._block_size - 1) // self._block_size

    def _sample(self, space: _SingleObjectiveTensorSpace) -> None:
        """Samples the next population with the CUDA Raw Kernel.

        The covariance adaptation is inherited from `CMAESTensor._adapt_distribution`.
        Set the class attribute `use_gemm = True` to sample with a cuBLAS GEMM
        (the `CMAESTensor` path) instead of the Raw Kernel, which is worth
        benchmarking for large dimensions: the kernel computes each output element
        with its own O(dim) loop and does not tile/reuse `transform` across agents.
        """

        if self.use_gemm:
            super()._sample(space)
            return

        xp = self._xp
        randn_buf = _randn(xp, (self.pop_size, self.dim), self.dtype)

        # The kernel needs a C-contiguous copy of `transform`; it only changes when
        # the covariance is decomposed, so the copy is refreshed on demand
        if self._transform_src is not self._transform:
            self._transform_c = xp.ascontiguousarray(self._transform, dtype=self.dtype)
            self._transform_src = self._transform

        mean_c = xp.ascontiguousarray(self.mean, dtype=self.dtype)

        sigma_c = xp.asarray(self._sigma, dtype=self.dtype).reshape(1)
        new_X_c = xp.empty((self.pop_size, self.dim), dtype=self.dtype)

        self._gen_kernel(
            (self._kernel_grid,),
            (self._block_size,),
            (
                new_X_c,
                randn_buf,
                mean_c,
                self._transform_c,
                sigma_c,
                xp.int32(self.dim),
                xp.int32(self._kernel_N),
            ),
        )

        space.X = new_X_c.reshape(space.X.shape)