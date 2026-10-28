"""L-SHADE: Success-History Based Adaptive Differential Evolution with Linear Population Size Reduction"""

from __future__ import annotations

import copy
import functools
import time
from typing import Dict, List, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function, Optimizer, TensorizedOptimizer
from opytimizer.core.space import _SingleObjectiveSpace, _SingleObjectiveTensorSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class LSHADE(Optimizer):
    """
    References:
        R. Tanabe and A. S. Fukunaga, "Improving the search performance of SHADE using linear population size reduction,"
        2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, China, 2014, pp. 1658-1665, doi: 10.1109/CEC.2014.6900380.
    """

    def __init__(
        self,
        params: Dict = None,
        MAX_NFE: int = 100,
        H: int = 6,
        p: float = 0.11,
        r_arc: float = 2.6,
        N_min: int = 4,
    ):

        logger.info("Overriding class: Optimizer -> L-SHADE (Default).")

        super().__init__()

        self.MAX_NFE = MAX_NFE
        self.H = H
        self.p = p
        self.r_arc = r_arc
        self.N_min = N_min

        self.build(params)

        self.A: List[np.ndarray] = []
        self.N_A = 0
        self.N_G = None
        self.N_init = None
        self.M_CR = None
        self.M_F = None
        self.k: int = None

        logger.info("Class overrided.")

    def compile(self, space: _SingleObjectiveSpace, **kwargs):
        self.N_G = space.n_agents
        self.N_init = space.n_agents
        self.M_CR = np.full(self.H, 0.5)
        self.M_F = np.full(self.H, 0.5)
        self.k = 0
        self.N_A = int(np.round(self.r_arc * self.N_G))
        self.A = []

    def _get_population_matrix(self, space: _SingleObjectiveSpace) -> np.ndarray:
        return np.array([ag.position.flatten() for ag in space.agents])

    def _mutate(
        self,
        F: np.ndarray,
        space: _SingleObjectiveSpace,
        X: Optional[np.ndarray] = None,
        fitness: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if X is None:
            X = self._get_population_matrix(space)
        N_G, D = X.shape

        pbest_bound = max(2, int(np.round(N_G * self.p)))
        if fitness is None:
            fitness = np.array([ag.fit for ag in space.agents])
        sorted_indices = np.argsort(fitness)
        p_best_indices_pool = sorted_indices[:pbest_bound]

        idx_pbest = np.random.choice(p_best_indices_pool, size=N_G)
        X_pbest = X[idx_pbest]

        idx_r1 = np.random.randint(0, N_G - 1, size=N_G)
        idx_r1 = np.where(idx_r1 >= np.arange(N_G), idx_r1 + 1, idx_r1)
        X_r1 = X[idx_r1]

        if len(self.A) > 0:
            A_matrix = np.array(self.A)
            X_union = np.vstack([X, A_matrix])
        else:
            X_union = X
        n_union = len(X_union)

        idx_r2 = np.random.randint(0, n_union - 2, size=N_G)
        exc1 = np.minimum(np.arange(N_G), idx_r1)
        exc2 = np.maximum(np.arange(N_G), idx_r1)
        idx_r2 = np.where(idx_r2 >= exc1, idx_r2 + 1, idx_r2)
        idx_r2 = np.where(idx_r2 >= exc2, idx_r2 + 1, idx_r2)
        X_r2 = X_union[idx_r2]

        F_col = F[:, np.newaxis]
        V = X + F_col * (X_pbest - X) + F_col * (X_r1 - X_r2)

        lb = np.array(space.lb).flatten()
        ub = np.array(space.ub).flatten()

        mask_low = V < lb
        V = np.where(mask_low, (lb + X) / 2.0, V)

        mask_high = V > ub
        V = np.where(mask_high, (ub + X) / 2.0, V)

        return V

    def _crossover(
        self,
        CR: np.ndarray,
        V: np.ndarray,
        space: _SingleObjectiveSpace,
        X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if X is None:
            X = self._get_population_matrix(space)
        N_G, D = X.shape

        rand_matrix = np.random.rand(N_G, D)
        crossover_mask = rand_matrix <= CR[:, np.newaxis]

        j_rand = np.random.randint(0, D, size=N_G)
        crossover_mask[np.arange(N_G), j_rand] = True

        U = np.where(crossover_mask, V, X)
        return U

    def update(self, space: _SingleObjectiveSpace, function: Function):
        selected_indices = np.random.randint(low=0, high=self.H, size=self.N_G)

        mu_CR = self.M_CR[selected_indices]
        mu_F = self.M_F[selected_indices]

        CR = np.random.normal(loc=mu_CR, scale=0.1)
        terminal_mask = np.isnan(mu_CR)
        CR = np.where(terminal_mask, 0.0, CR)
        CR = np.clip(CR, 0.0, 1.0)

        # Cauchy(mu_F, 0.1) samples by inverse-CDF sampling on NumPy uniforms.
        F = mu_F + 0.1 * np.tan(np.pi * np.random.uniform(size=self.N_G) - np.pi / 2.0)
        invalid_mask = F <= 0.0
        while np.any(invalid_mask):
            num_invalid = int(np.sum(invalid_mask))
            F[invalid_mask] = mu_F[invalid_mask] + 0.1 * np.tan(
                np.pi * np.random.uniform(size=num_invalid) - np.pi / 2.0
            )
            invalid_mask = F <= 0.0
        F = np.clip(F, None, 1.0)

        # Population matrix and fitness are built once and shared by mutation,
        # crossover and the archive update below (positions do not change until the
        # survivors are written back).
        X_pop = self._get_population_matrix(space)
        f_X = np.array([ag.fit for ag in space.agents])

        V = self._mutate(F, space, X_pop, f_X)
        U = self._crossover(CR, V, space, X_pop)

        f_X = np.array([ag.fit for ag in space.agents])

        f_U = np.zeros(self.N_G)
        for i, ag in enumerate(space.agents):
            trial_position = U[i].reshape(ag.position.shape)
            f_U[i] = function(trial_position).squeeze()

        survival_mask = f_U <= f_X
        success_mask = f_U < f_X

        S_CR = CR[success_mask]
        S_F = F[success_mask]
        DELTA_F = np.abs(f_X[success_mask] - f_U[success_mask])

        X_old = X_pop

        for i in np.flatnonzero(survival_mask):
            ag = space.agents[i]
            ag.position = U[i].reshape(ag.position.shape)
            ag.fit = f_U[i]

            if success_mask[i]:
                self.A.append(X_old[i])

            if ag.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(ag.position)
                space.best_agent.fit = copy.deepcopy(ag.fit)
                space.best_agent.ts = int(time.time())

        if len(self.A) > self.N_A:
            indices_to_keep = np.random.choice(
                len(self.A), size=self.N_A, replace=False
            )
            self.A = [self.A[idx] for idx in indices_to_keep]

        if len(S_CR) > 0:
            sum_delta = np.sum(DELTA_F)
            weights = DELTA_F / (sum_delta if sum_delta > 0 else 1e-10)

            # Weighted Arithmetic Mean for M_CR
            if np.isnan(self.M_CR[self.k]) or np.max(S_CR) == 0:
                self.M_CR[self.k] = np.nan
            else:
                self.M_CR[self.k] = np.sum(weights * S_CR)

            # Weighted Lehmer Mean for M_F
            self.M_F[self.k] = np.sum(weights * (S_F**2)) / np.sum(weights * S_F)

            self.k = (self.k + 1) % self.H

        N_target = int(
            np.round(
                ((self.N_min - self.N_init) / self.MAX_NFE) * function.n_calls
                + self.N_init
            )
        )
        N_target = max(self.N_min, N_target)

        if N_target < self.N_G:
            space.agents.sort(key=lambda ag: ag.fit)
            space.agents = space.agents[:N_target]

            self.N_G = N_target
            self.N_A = int(np.round(self.r_arc * self.N_G))

            if len(self.A) > self.N_A:
                self.A = self.A[: self.N_A]

    def evaluate(self, space, function):
        for agent in space.agents:
            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

        self.evaluate = lambda: None


@functools.lru_cache(maxsize=None)
def _get_lshade_trial_kernel(xp):
    """Fused CUDA kernel: current-to-pbest/1 mutation + bound repair + binomial crossover.

    For every (agent, variable) element it gathers x, x_pbest, x_r1 and x_r2
    straight from the population / archive buffers, builds the mutant, repairs
    bound violations and applies the crossover, writing the trial vector `u`. This
    replaces three row gathers, an archive concatenation,

        X: (N, nf) population
        A : (cap + 1, nf) archive buffer
        ipb/ir1: (N, 1)  row of x_pbest / x_r1 in X
        ir2: (N, 1)  row of x_r2 in the union [X; A] (>= n_pop means archive)
        f, cr, jr: (N, 1)  per-agent F, CR and forced crossover column
        lb, ub: (1, nf) bounds
        rnd : (N, nf) uniform samples
    """

    return xp.ElementwiseKernel(
        "raw T X, raw T A, int32 ipb, int32 ir1, int32 ir2, T f, T lb, T ub, "
        "T rnd, T cr, int32 jr, int32 nf, int32 n_pop",
        "T u",
        """
        int row = i / nf;
        int col = i - row * nf;
        T x = X[row * nf + col];
        T xpb = X[ipb * nf + col];
        T xr1 = X[ir1 * nf + col];
        T xr2 = (ir2 < n_pop) ? X[ir2 * nf + col] : A[(ir2 - n_pop) * nf + col];
        T v = x + f * (xpb - x) + f * (xr1 - xr2);
        if (v < lb) v = (lb + x) / (T)2;
        if (v > ub) v = (ub + x) / (T)2;
        u = (rnd <= cr || col == jr) ? v : x;
        """,
        "lshade_trial_vector",
    )


def _uniform(xp, size, dtype):
    """Uniform [0, 1) samples directly in `dtype`"""
    if xp.__name__ == "cupy":
        return xp.random.random_sample(size, dtype=dtype)

    return xp.random.random_sample(size).astype(dtype)


def _normal(xp, size, dtype):
    """Standard-normal samples directly in `dtype` (see `_uniform`)."""
    if xp.__name__ == "cupy":
        return xp.random.standard_normal(size, dtype=dtype)

    return xp.random.standard_normal(size).astype(dtype)


class LSHADETensor(Optimizer, TensorizedOptimizer):
    """Agnostic Tensorized L-SHADE Implementation (NumPy/CuPy)."""

    def __init__(
        self,
        params: Dict = None,
        MAX_NFE: int = 100,
        H: int = 6,
        p: float = 0.11,
        r_arc: float = 2.6,
        N_min: int = 4,
    ):

        logger.info("Overriding class: Optimizer -> L-SHADE (Tensor).")

        super().__init__()

        self.MAX_NFE = MAX_NFE
        self.H = H
        self.p = p
        self.r_arc = r_arc
        self.N_min = N_min

        # External archive A: fixed-capacity device buffer with `_A_cap + 1` rows
        # (the extra one is a "trash" row, see `update`) and a device-resident count.
        # `self.A` exposes the valid members as a read-only property.
        self._A_buf = None
        self._A_n = None
        self._A_cap = 0
        self.N_A = 0
        self.N_G = None
        self.N_init = None
        self.M_CR = None
        self.M_F = None
        self.k = None  # device int32 scalar: next slot of the historical memory
        self.dtype = None

        self._xp = None  # concrete array module (numpy / cupy), NOT the proxy
        self._use_fused = False  # fused CUDA trial-vector kernel available?

        self.global_best_position = None
        self.global_best_fit = float("inf")

        self.build(params)
        logger.info("Class overrided.")

    @property
    def A(self):
        """Current members of the external archive, shape (|A|, *feature_shape)"""
        if self._A_buf is None:
            return None

        return self._A_buf[: int(self._A_n)]

    def compile(self, space: _SingleObjectiveTensorSpace, **kwargs):
        xp = space.env.xp
        self._xp = xp.__wrapped__
        self.dtype = xp.dtype(space.env.dtype)
        feat_shape = space.X.shape[1:]
        n_features = int(np.prod(feat_shape))

        self.N_G = space.n_agents
        self.N_init = space.n_agents
        self.M_CR = xp.full(self.H, 0.5, dtype=self.dtype)
        self.M_F = xp.full(self.H, 0.5, dtype=self.dtype)
        self.k = xp.zeros((), dtype=xp.int32)
        self.N_A = int(np.round(self.r_arc * self.N_G))

        # The archive capacity only ever shrinks (N_A follows the population size),
        # so a buffer sized for the initial capacity is enough for the whole run.
        self._A_cap = self.N_A
        self._A_buf = xp.zeros((self._A_cap + 1,) + feat_shape, dtype=self.dtype)
        self._A_n = xp.zeros((), dtype=xp.int32)

        # Bounds and column indices are constant: flatten/convert them once.
        self._lb2 = xp.asarray(space.lb, dtype=self.dtype).reshape(1, n_features)
        self._ub2 = xp.asarray(space.ub, dtype=self.dtype).reshape(1, n_features)
        self._cols = xp.arange(n_features, dtype=xp.int32).reshape(1, n_features)

        self._use_fused = bool(space.env.use_cuda)

        self.global_best_position = xp.zeros(feat_shape, dtype=self.dtype)
        self.global_best_fit = xp.asarray(xp.inf, dtype=self.dtype)

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function):
        xp = self._xp
        space.F = function(space.X, xp=xp)
        min_idx = xp.argmin(space.F)
        space.best_agent.position = space.X[min_idx].copy()
        space.best_agent.fit = space.F[min_idx].copy()
        space.best_agent.ts = int(time.time())

        self.global_best_position = space.best_agent.position.copy()
        self.global_best_fit = space.best_agent.fit.copy()

        self.evaluate = lambda: None

    def _sample_parameters(self, xp, N_G):
        """Draws CR_i ~ N(mu_CR, 0.1) clipped to [0, 1] and F_i ~ Cauchy(mu_F, 0.1)
        conditioned on F_i > 0 and clipped to 1"""

        dtype = self.dtype

        selected = xp.random.randint(0, self.H, size=N_G)
        mu_CR = self.M_CR[selected]
        mu_F = self.M_F[selected]

        CR = mu_CR + 0.1 * _normal(xp, N_G, dtype)
        CR = xp.where(xp.isnan(mu_CR), 0.0, CR)
        CR = xp.clip(CR, 0.0, 1.0)

        u_min = 0.5 - xp.arctan(mu_F / 0.1) / np.pi
        u = u_min + (1.0 - u_min) * _uniform(xp, N_G, dtype)
        F = mu_F + 0.1 * xp.tan(np.pi * (u - 0.5))

        # The floor only guards against rounding at the edge of the interval.
        F = xp.clip(F, float(np.finfo(dtype).eps), 1.0)

        return CR, F

    def _trial_vectors(self, space, CR, F, n_features):
        """Mutation (current-to-pbest/1 with archive) + bound repair + binomial crossover."""

        xp = self._xp
        dtype = self.dtype
        N_G = space.X.shape[0]

        X2 = space.X.reshape(N_G, n_features)
        A2 = self._A_buf.reshape(self._A_cap + 1, n_features)
        arange = xp.arange(N_G)

        # p-best pool: the `pbest_bound` best individuals.
        pbest_bound = max(2, int(np.round(N_G * self.p)))
        pool = xp.argsort(space.F)[:pbest_bound]
        idx_pbest = pool[xp.random.randint(0, pbest_bound, size=N_G)]

        idx_r1 = xp.random.randint(0, N_G - 1, size=N_G)
        idx_r1 = xp.where(idx_r1 >= arange, idx_r1 + 1, idx_r1)

        n_union = N_G + self._A_n

        idx_r2 = (_uniform(xp, N_G, dtype) * (n_union - 2).astype(dtype)).astype(
            xp.int32
        )
        idx_r2 = xp.minimum(idx_r2, n_union - 3)
        exc1 = xp.minimum(arange, idx_r1)
        exc2 = xp.maximum(arange, idx_r1)
        idx_r2 = xp.where(idx_r2 >= exc1, idx_r2 + 1, idx_r2)
        idx_r2 = xp.where(idx_r2 >= exc2, idx_r2 + 1, idx_r2)

        rnd = _uniform(xp, (N_G, n_features), dtype)
        j_rand = xp.random.randint(0, n_features, size=N_G).astype(xp.int32)

        if self._use_fused:
            try:
                U = xp.empty((N_G, n_features), dtype=dtype)
                _get_lshade_trial_kernel(xp)(
                    X2,
                    A2,
                    idx_pbest.astype(xp.int32)[:, xp.newaxis],
                    idx_r1.astype(xp.int32)[:, xp.newaxis],
                    idx_r2.astype(xp.int32)[:, xp.newaxis],
                    F[:, xp.newaxis],
                    self._lb2,
                    self._ub2,
                    rnd,
                    CR[:, xp.newaxis],
                    j_rand[:, xp.newaxis],
                    n_features,
                    N_G,
                    U,
                )
                return U
            except Exception as exc:  # pragma: no cover - depends on CUDA runtime
                logger.warning(
                    "LSHADETensor: fused CUDA kernel unavailable (%s). "
                    "Falling back to the portable implementation.",
                    exc,
                )
                self._use_fused = False

        # Portable path (numpy, or cupy without the fused kernel).
        X_pbest = X2[idx_pbest]
        X_r1 = X2[idx_r1]
        X_r2 = xp.where(
            (idx_r2 < N_G)[:, xp.newaxis],
            X2[xp.minimum(idx_r2, N_G - 1)],
            A2[xp.maximum(idx_r2 - N_G, 0)],
        )

        F_col = F[:, xp.newaxis]
        V = X2 + F_col * (X_pbest - X2) + F_col * (X_r1 - X_r2)
        V = xp.where(V < self._lb2, (self._lb2 + X2) / 2.0, V)
        V = xp.where(V > self._ub2, (self._ub2 + X2) / 2.0, V)

        take_mutant = (rnd <= CR[:, xp.newaxis]) | (self._cols == j_rand[:, xp.newaxis])
        return xp.where(take_mutant, V, X2)

    def _insert_into_archive(self, space, success):
        """Adds the PARENTS of the successful trials to the external archive."""

        xp = self._xp
        N_G = space.X.shape[0]
        cap = self.N_A

        success_i = success.astype(xp.int32)
        offset = xp.cumsum(success_i).astype(xp.int32) - 1  # slot among the successes
        n_success = offset[-1] + 1

        append_slot = self._A_n + offset
        random_slot = xp.minimum(
            (_uniform(xp, N_G, self.dtype) * cap).astype(xp.int32), cap - 1
        )
        slot = xp.where(append_slot < cap, append_slot, random_slot)
        slot = xp.where(success, slot, self._A_cap)  # non-successes -> trash row

        self._A_buf[slot] = space.X
        self._A_n = xp.minimum(self._A_n + n_success, cap).astype(xp.int32)

    def _update_memory(self, success, CR, F, f_old, f_new):
        """Weighted Lehmer/arithmetic update of the historical memory slot `k`."""

        xp = self._xp
        H = self.H

        delta = xp.where(success, xp.abs(f_old - f_new), 0.0)
        sum_delta = xp.sum(delta)
        weights = delta / xp.where(sum_delta < 1e-10, 1e-10, sum_delta)

        has_success = xp.any(success)
        max_success_cr = xp.max(xp.where(success, CR, 0.0))

        # Weighted arithmetic mean for M_CR (NaN is the "terminal" value).
        current_cr = self.M_CR[self.k]
        new_cr = xp.where(
            xp.isnan(current_cr) | (max_success_cr == 0),
            xp.nan,
            xp.sum(weights * CR),
        )

        # Weighted Lehmer mean for M_F.
        sum_w_sf = xp.sum(weights * F)
        new_f = xp.where(
            sum_w_sf < 1e-10,
            sum_w_sf,
            xp.sum(weights * F * F) / xp.where(sum_w_sf < 1e-10, 1.0, sum_w_sf),
        )

        slot_mask = (xp.arange(H, dtype=xp.int32) == self.k) & has_success
        self.M_CR = xp.where(slot_mask, new_cr, self.M_CR).astype(self.dtype)
        self.M_F = xp.where(slot_mask, new_f, self.M_F).astype(self.dtype)
        self.k = xp.where(has_success, (self.k + 1) % H, self.k).astype(xp.int32)

    def update(self, space: _SingleObjectiveTensorSpace, function: Function):
        xp = self._xp
        N_G = space.X.shape[0]
        feat_shape = space.X.shape[1:]
        n_features = int(np.prod(feat_shape))

        CR, F = self._sample_parameters(xp, N_G)
        U = self._trial_vectors(space, CR, F, n_features)

        # Evaluation and selection
        f_U = function(U.reshape(space.X.shape), xp=xp)

        survival_mask = f_U <= space.F
        success_mask = f_U < space.F

        # These three use the OLD population / fitness, so they run before the
        # in-place write-back below.
        self._insert_into_archive(space, success_mask)
        self._update_memory(success_mask, CR, F, space.F, f_U)

        # In-place masked copies
        survival_b = survival_mask.reshape((N_G,) + (1,) * len(feat_shape))
        xp.copyto(space.X, U.reshape(space.X.shape), where=survival_b)
        xp.copyto(space.F, f_U, where=survival_mask)

        min_idx = xp.argmin(space.F)
        min_fit = space.F[min_idx]
        improved = min_fit < self.global_best_fit
        self.global_best_position = xp.where(
            improved, space.X[min_idx], self.global_best_position
        )
        self.global_best_fit = xp.where(improved, min_fit, self.global_best_fit).astype(
            self.dtype
        )
        space.best_agent.position = self.global_best_position.reshape(
            space.best_agent.position.shape
        )
        space.best_agent.fit = self.global_best_fit
        space.best_agent.ts = int(time.time())

        N_target = int(
            np.round(
                ((self.N_min - self.N_init) / self.MAX_NFE) * function.n_calls
                + self.N_init
            )
        )
        N_target = max(self.N_min, N_target)

        if N_target < self.N_G:
            keep = xp.argsort(space.F)[:N_target]
            space.X = space.X[keep]
            space.F = space.F[keep]

            space.agents = space.agents[:N_target]
            space.n_agents = N_target

            self.N_G = N_target
            self.N_A = int(np.round(self.r_arc * self.N_G))

            # Shrinking the capacity just truncates the (device-side) member count.
            self._A_n = xp.minimum(self._A_n, self.N_A).astype(xp.int32)
