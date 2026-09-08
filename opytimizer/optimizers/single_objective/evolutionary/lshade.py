"""L-SHADE: Success-History Based Adaptive Differential Evolution with Linear Population Size Reduction"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import cauchy

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
        H: int = 100,
        p: float = 0.11,
        f_arc: float = 2.6,
    ):

        logger.info("Overriding class: Optimizer -> L-SHADE (Default).")

        super().__init__()

        self.MAX_NFE = MAX_NFE
        self.H = H
        self.p = p
        self.f_arc = f_arc

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
        self.N_A = int(np.round(self.f_arc * self.N_G))
        self.A = []

    def _get_population_matrix(self, space: _SingleObjectiveSpace) -> np.ndarray:
        return np.array([ag.position.flatten() for ag in space.agents])

    def _mutate(self, F: np.ndarray, space: _SingleObjectiveSpace) -> np.ndarray:
        N_G = self.N_G
        X = self._get_population_matrix(space)
        N_G, D = X.shape

        pbest_bound = max(2, int(np.round(N_G * self.p)))
        _fitness = np.array([ag.fit for ag in space.agents])
        sorted_indices = np.argsort(_fitness)
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
        self, CR: np.ndarray, V: np.ndarray, space: _SingleObjectiveSpace
    ) -> np.ndarray:
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

        F = cauchy.rvs(loc=mu_F, scale=0.1, size=self.N_G)
        invalid_mask = F <= 0.0
        while np.any(invalid_mask):
            num_invalid = np.sum(invalid_mask)
            new_samples = cauchy.rvs(
                loc=mu_F[invalid_mask], scale=0.1, size=num_invalid
            )
            F[invalid_mask] = new_samples
            invalid_mask = F <= 0.0
        F = np.clip(F, None, 1.0)

        V = self._mutate(F, space)
        U = self._crossover(CR, V, space)

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

        X_old = self._get_population_matrix(space)
        for i, ag in enumerate(space.agents):
            if survival_mask[i]:
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

        N_min = 4
        N_target = int(
            np.round(
                ((N_min - self.N_init) / self.MAX_NFE) * function.n_calls + self.N_init
            )
        )
        N_target = max(N_min, N_target)

        if N_target < self.N_G:
            space.agents.sort(key=lambda ag: ag.fit)
            space.agents = space.agents[:N_target]

            self.N_G = N_target
            self.N_A = int(np.round(self.f_arc * self.N_G))

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


class LSHADETensor(Optimizer, TensorizedOptimizer):
    """Agnostic Tensorized L-SHADE Implementation (NumPy/CuPy)."""

    def __init__(
        self,
        params: Dict = None,
        MAX_NFE: int = 100,
        H: int = 100,
        p: float = 0.11,
        f_arc: float = 2.6,
    ):

        logger.info("Overriding class: Optimizer -> L-SHADE (Tensor).")

        super().__init__()

        self.MAX_NFE = MAX_NFE
        self.H = H
        self.p = p
        self.f_arc = f_arc

        self.A = None
        self.N_A = 0
        self.N_G = None
        self.N_init = None
        self.M_CR = None
        self.M_F = None
        self.k: int = None
        self.dtype = None

        self.global_best_position = None
        self.global_best_fit = float("inf")

        self.build(params)
        logger.info("Class overrided.")

    def compile(self, space: _SingleObjectiveTensorSpace, **kwargs):
        xp = space.env.xp
        self.dtype = xp.dtype(space.env.dtype)

        self.N_G = space.n_agents
        self.N_init = space.n_agents
        self.M_CR = xp.full(self.H, 0.5, dtype=self.dtype)
        self.M_F = xp.full(self.H, 0.5, dtype=self.dtype)
        self.k = 0
        self.N_A = int(np.round(self.f_arc * self.N_G))
        self.A = xp.zeros((0,) + space.X.shape[1:], dtype=self.dtype)

        self.global_best_position = xp.zeros(space.X.shape[1:], dtype=self.dtype)
        self.global_best_fit = float("inf")

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function):
        xp = space.env.xp
        space.F = function(space.X, xp=xp)
        min_idx = xp.argmin(space.F)
        space.best_agent.position = space.X[min_idx].copy()
        space.best_agent.fit = space.F[min_idx].copy()
        space.best_agent.ts = int(time.time())

        self.global_best_position = space.best_agent.position.copy()
        self.global_best_fit = space.best_agent.fit.copy()

        self.evaluate = lambda: None

    def update(self, space: _SingleObjectiveTensorSpace, function: Function):
        xp = space.env.xp
        N_G = space.X.shape[0]
        feat_shape = space.X.shape[1:]
        n_features = int(np.prod(feat_shape))
        shape_broadcast = (N_G,) + (1,) * (space.X.ndim - 1)

        # Parameter Generation
        selected_indices = xp.random.randint(0, self.H, size=N_G)
        mu_CR = self.M_CR[selected_indices]
        mu_F = self.M_F[selected_indices]

        CR = xp.random.normal(loc=mu_CR, scale=0.1).astype(self.dtype)
        CR = xp.where(xp.isnan(mu_CR), xp.asarray(0.0, dtype=self.dtype), CR)
        CR = xp.clip(
            CR, xp.asarray(0.0, dtype=self.dtype), xp.asarray(1.0, dtype=self.dtype)
        )

        # Cauchy distribution sampling via Inverse Transform Sampling
        u = xp.random.uniform(0.0, 1.0, size=N_G).astype(self.dtype)
        pi_val = xp.asarray(np.pi, dtype=self.dtype)
        F = mu_F + xp.asarray(0.1, dtype=self.dtype) * xp.tan(
            pi_val * (u - xp.asarray(0.5, dtype=self.dtype))
        )
        invalid_mask = F <= 0.0
        while xp.any(invalid_mask):
            num_invalid = int(xp.sum(invalid_mask))
            u_sub = xp.random.uniform(0.0, 1.0, size=num_invalid).astype(self.dtype)
            F[invalid_mask] = mu_F[invalid_mask] + xp.asarray(
                0.1, dtype=self.dtype
            ) * xp.tan(pi_val * (u_sub - xp.asarray(0.5, dtype=self.dtype)))
            invalid_mask = F <= 0.0
        F = xp.clip(F, a_min=None, a_max=xp.asarray(1.0, dtype=self.dtype))

        # Mutation
        pbest_bound = max(2, int(np.round(N_G * self.p)))
        sorted_indices = xp.argsort(space.F)
        pbest_pool = sorted_indices[:pbest_bound]
        idx_pbest = pbest_pool[xp.random.randint(0, pbest_bound, size=N_G)]
        X_pbest = space.X[idx_pbest]

        idx_r1 = xp.random.randint(0, N_G - 1, size=N_G)
        idx_r1 = xp.where(idx_r1 >= xp.arange(N_G), idx_r1 + 1, idx_r1)
        X_r1 = space.X[idx_r1]

        if self.A.shape[0] > 0:
            X_union = xp.concatenate([space.X, self.A], axis=0)
        else:
            X_union = space.X
        n_union = X_union.shape[0]

        idx_r2 = xp.random.randint(0, n_union - 2, size=N_G)
        exc1 = xp.minimum(xp.arange(N_G), idx_r1)
        exc2 = xp.maximum(xp.arange(N_G), idx_r1)
        idx_r2 = xp.where(idx_r2 >= exc1, idx_r2 + 1, idx_r2)
        idx_r2 = xp.where(idx_r2 >= exc2, idx_r2 + 1, idx_r2)
        X_r2 = X_union[idx_r2]

        F_col = F.reshape(shape_broadcast)
        V = space.X + F_col * (X_pbest - space.X) + F_col * (X_r1 - X_r2)

        # Explicitly reshape bounds to match feature dimensions
        lb = xp.asarray(space.lb, dtype=self.dtype).reshape((1,) + feat_shape)
        ub = xp.asarray(space.ub, dtype=self.dtype).reshape((1,) + feat_shape)
        V = xp.where(V < lb, (lb + space.X) / 2.0, V)
        V = xp.where(V > ub, (ub + space.X) / 2.0, V)

        # Crossover
        CR_col = CR.reshape(shape_broadcast)
        rand_matrix = xp.random.uniform(0.0, 1.0, size=space.X.shape).astype(self.dtype)
        crossover_mask = rand_matrix <= CR_col

        j_rand = xp.random.randint(0, n_features, size=N_G)
        crossover_mask_flat = crossover_mask.reshape((N_G, n_features))
        crossover_mask_flat[xp.arange(N_G), j_rand] = True
        crossover_mask = crossover_mask_flat.reshape(space.X.shape)

        U = xp.where(crossover_mask, V, space.X)

        # Evaluation and Selection
        f_U = function(U, xp=xp)
        if not isinstance(f_U, xp.ndarray):
            f_U = xp.asarray(f_U, dtype=self.dtype)

        survival_mask = f_U <= space.F
        success_mask = f_U < space.F

        S_CR = CR[success_mask]
        S_F = F[success_mask]
        DELTA_F = xp.abs(space.F[success_mask] - f_U[success_mask])

        if xp.any(success_mask):
            self.A = xp.concatenate([self.A, space.X[success_mask]], axis=0)

        survival_broadcast = survival_mask.reshape(shape_broadcast)
        space.X = xp.where(survival_broadcast, U, space.X)
        space.F = xp.where(survival_mask, f_U, space.F)

        min_fit = xp.min(space.F)
        if min_fit < space.best_agent.fit:
            min_idx = xp.argmin(space.F)
            space.best_agent.position = space.X[min_idx].copy()
            space.best_agent.fit = space.F[min_idx].copy()
            space.best_agent.ts = int(time.time())

        self.global_best_position = space.best_agent.position.copy()
        self.global_best_fit = space.best_agent.fit.copy()
        # Truncate archive if capacity exceeded
        if self.A.shape[0] > self.N_A:
            perm = xp.random.permutation(self.A.shape[0])[: self.N_A]
            self.A = self.A[perm]

        # Historical Memory Update
        if S_CR.shape[0] > 0:
            sum_delta = xp.sum(DELTA_F)
            weights = DELTA_F / xp.where(sum_delta < 1e-10, 1e-10, sum_delta)

            # Weighted Arithmetic Mean for M_CR
            if xp.isnan(self.M_CR[self.k]) or xp.max(S_CR) == 0:
                self.M_CR[self.k] = float("nan")
            else:
                self.M_CR[self.k] = xp.sum(weights * S_CR)

            # Weighted Lehmer Mean for M_F
            sum_w_sf = xp.sum(weights * S_F)
            if sum_w_sf < 1e-10:
                self.M_F[self.k] = xp.sum(weights * S_F)
            else:
                self.M_F[self.k] = xp.sum(weights * (S_F**2)) / sum_w_sf

            self.k = (self.k + 1) % self.H

        # Linear Population Size Reduction (LPSR)
        N_min = 4
        N_target = int(
            np.round(
                ((N_min - self.N_init) / self.MAX_NFE) * function.n_calls + self.N_init
            )
        )
        N_target = max(N_min, N_target)

        if N_target < self.N_G:
            sort_idx = xp.argsort(space.F)
            space.X = space.X[sort_idx[:N_target]]
            space.F = space.F[sort_idx[:N_target]]

            space.agents = space.agents[:N_target]
            space.n_agents = N_target

            self.N_G = N_target
            self.N_A = int(np.round(self.f_arc * self.N_G))
            if self.A.shape[0] > self.N_A:
                self.A = self.A[: self.N_A]
