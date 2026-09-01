"""
CMA-ES — Covariance Matrix Adaptation Evolution Strategy
"""

from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function, Optimizer
from opytimizer.core.space import _SingleObjectiveSpace, _SingleObjectiveTensorSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


"""
    Reference
    ---------
    Hansen, N. (2023). The CMA Evolution Strategy: A Tutorial.
    arXiv:1604.00772v2 [cs.LG]. https://arxiv.org/abs/1604.00772
"""

class _CMAESMixin:
    """
    Pure-math helpers shared by the CPU and CUDA backends.
    """

    def _init_state_arrays(self, n: int, xp):
        """Allocate evolution-path vectors and covariance-matrix factors."""
        self.ps = xp.zeros(n)
        self.pc = xp.zeros(n)
        self.C = xp.eye(n)
        self.B = xp.eye(n)
        self.D = xp.eye(n)

    def _compute_hyperparams(self, n: int, xp):
        # Population / parent sizes  (Table 1 / Appendix B)
        #self.lam = 4 + int(math.floor(3 * math.log(n)))
        self.mu = self.lam // 2

        # (Eq. 49 / 53)
        w_raw   = xp.log(self.lam / 2.0 + 0.5) - xp.log(xp.arange(1, self.lam + 1))
        w_pos   = w_raw[: self.mu]
        w_pos_n = w_pos / w_pos.sum() # positive weights sum to 1

        # (Eq. 8)
        self.mueff = float(1.0 / xp.sum(w_pos_n ** 2))

        w_neg_raw = w_raw[self.mu:]
        mueff_neg = (
            float(w_neg_raw.sum() ** 2 / xp.sum(w_neg_raw ** 2))
            if len(w_neg_raw) else 1.0
        )

        # (Eq. 55)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.ds = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0)
            + self.cs
        )

        # (Eq. 56-58)
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        alpha_cov = 2.0
        self.cmu = min(
            1.0 - self.c1,
            alpha_cov
            * (self.mueff - 2.0 + 1.0 / self.mueff)
            / ((n + 2.0) ** 2 + alpha_cov * self.mueff / 2.0),
        )

        # (Eq. 50-52)
        alpha_mu = 1.0 + self.c1 / self.cmu  if self.cmu > 0 else 1.0
        alpha_mueff = 1.0 + 2.0 * mueff_neg / (self.mueff + 2.0)
        alpha_posdef = (
            (1.0 - self.c1 - self.cmu) / (n * self.cmu) if self.cmu > 0 else 1.0
        )
        alpha_neg = min(alpha_mu, alpha_mueff, alpha_posdef)

        # Final weight vector  (Eq. 53)
        w_neg_sum = xp.abs(w_neg_raw).sum() if len(w_neg_raw) else 1.0
        w_neg_sc = alpha_neg / w_neg_sum * w_neg_raw
        self.weights = xp.concatenate([w_pos_n, w_neg_sc])

        # Cache quantities that never change after compile()
        self.sum_weights = float(xp.sum(self.weights))
        self.w_mu = self.weights[: self.mu]

        # E||N(0,I)||  (Appendix A)
        self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

    # (Eq. 38-40)
    def _sample_population(self, xp) -> tuple:
        Z = xp.random.randn(self.lam, self.n)  # Eq. 38
        BD = self.B @ self.D  # Eq. 39
        Y = Z @ BD.T #  Eq. 39
        X = self.m + self.sigma * Y  #  Eq. 40
        return X, Y, Z

    # Bounds repair (Appendix B.5)
    def _clip_to_bounds(self, X, lb, ub, xp):
        return xp.clip(X, lb, ub)

    #  (Eq. 42)
    def _update_mean(self, Y_sorted, xp):
        yw = self.w_mu @ Y_sorted[: self.mu] # Eq. 42
        self.m = self.m + self.sigma * yw
        return yw

    # (Eq. 43-44)
    def _update_step_size(self, yw, Z_sorted, xp) -> float:
        zw = self.w_mu @ Z_sorted[: self.mu] # (n,)
        invsqrtC_yw = self.B @ zw

        # (Eq. 43)
        self.ps = (
            (1.0 - self.cs) * self.ps
            + float(math.sqrt(self.cs * (2.0 - self.cs) * self.mueff))
            * invsqrtC_yw
        )

        # (Eq. 44)
        ps_norm = float(xp.linalg.norm(self.ps))
        self.sigma *= float(math.exp((self.cs / self.ds) * (ps_norm / self.chiN - 1.0)))
        return ps_norm
    
    # Stall indicator  (Appendix A / Fig. 6)
    def _hsig(self, ps_norm: float, xp) -> int:
        correction = float(math.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * self.g)))
        threshold = (1.4 + 2.0 / (self.n + 1.0)) * self.chiN
        return int((ps_norm / correction) < threshold)

    # (Eq. 45-47)
    def _update_covariance(self, yw, Y_sorted, hsig: int, xp):
        # (Eq. 45)
        self.pc = (
            (1.0 - self.cc) * self.pc
            + hsig
            * float(math.sqrt(self.cc * (2.0 - self.cc) * self.mueff))
            * yw
        )

        # (Eq. 46)
        BtY = Y_sorted @ self.B  
        d_diag = xp.diag(self.D) 
        DinvBtY = BtY / d_diag 
        mahal_sq = xp.sum(DinvBtY ** 2, axis=1)  

        w_circle= xp.where(
            self.weights >= 0,
            self.weights,
            self.weights * self.n / (mahal_sq + 1e-20), # Eq. 46
        )

        # (Eq. 47)
        delta_hsig = (1.0 - hsig) * self.cc * (2.0 - self.cc)
        decay = (
            1.0
            + self.c1 * delta_hsig
            - self.c1
            - self.cmu * self.sum_weights  
        )

        # (Eq. 45)
        rank1 = xp.outer(self.pc, self.pc)

        # (Eq. 47)
        rankmu = (Y_sorted * w_circle[:, None]).T @ Y_sorted

        self.C = decay * self.C + self.c1 * rank1 + self.cmu * rankmu

        # Enforce symmetry
        self.C = xp.triu(self.C) + xp.triu(self.C, 1).T

    # Eigendecomposition (lazy)  (Appendix B.2)
    def _update_eigensystem(self, total_evals: int, xp):
        c_sum = self.c1 + self.cmu
        interval = max(1, int(math.floor(1.0 / (10.0 * self.n * c_sum))))

        if self.g - self.eigeneval >= interval:
            self.eigeneval = self.g

            eigenvalues, self.B = xp.linalg.eigh(self.C)  # ascending order
            eigenvalues = xp.maximum(eigenvalues, 1e-20) # numerical guard
            self.D = xp.diag(xp.sqrt(eigenvalues))


class CMAES(Optimizer, _CMAESMixin):
    """CMA-ES — Liner Execution."""

    def __init__(
        self,
        params: Dict = None,
        sigma0: float = 0.3,
        **kwds,
    ):
        logger.info("Overriding class: Optimizer -> CMA-ES (Default).")
        super().__init__()

        self.sigma0 = sigma0
        self.build(params)

        # Dimension
        self.n: int = None
        self.lam: int = None
        self.mu: int = None
        self.weights = None
        self.mueff: float = None

        # Step-size control
        self.cs: float = None
        self.ds: float = None

        # Covariance-matrix learning rates
        self.cc: float = None
        self.c1: float = None
        self.cmu: float = None

        # Distribution state
        self.m: np.ndarray = None
        self.sigma: float = None
        self.C: np.ndarray = None
        self.B: np.ndarray = None
        self.D: np.ndarray = None
        self.ps: np.ndarray = None
        self.pc: np.ndarray = None
        self.chiN: float = None
        self.g: int = None
        self.eigeneval: int = None
        
        self._lb = None
        self._ub = None

        logger.info("Class overrided.")

    # ------------------------------------------------------------------

    def compile(self, space: _SingleObjectiveSpace, **kwargs):
        xp = np  
        n = len(space.lb)
        self.n = n
        self.lam = space.n_agents
        self._compute_hyperparams(n, xp)

      
        self.m = np.mean([ag.position.squeeze() for ag in space.agents], axis=0)
        self.sigma = float(self.sigma0)
        self.g = 0
        self.eigeneval = 0

        self._init_state_arrays(n, xp)
        
       
        self._lb = space.lb
        self._ub = space.ub

        logger.debug(
            "CMA-ES (CPU) compiled: n=%d  lambda=%d  µ=%d  µ_eff=%.2f  "
            "c1=%.4f  cµ=%.4f  cc=%.4f  cs=%.4f  ds=%.4f",
            n, self.lam, self.mu, self.mueff,
            self.c1, self.cmu, self.cc, self.cs, self.ds,
        )

    # ------------------------------------------------------------------

    def evaluate(self, space: _SingleObjectiveSpace, function: Function):
        for agent in space.agents:
            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = agent.position.copy()
                space.best_agent.fit = agent.fit
                space.best_agent.ts = int(time.time())

        
        self.evaluate = lambda: None

    # ------------------------------------------------------------------

    def update(self, space: _SingleObjectiveSpace, function: Function):
        xp = np
        self.g += 1

        # (Eq. 38-40)
        X, Y, Z = self._sample_population(xp)
        X = self._clip_to_bounds(X, self._lb, self._ub, xp)

        # Evaluate — one call per candidate 
        f_vals = np.empty(self.lam)
        for k in range(self.lam):
            f_vals[k] = float(function(X[k].reshape(-1, 1)))

        # Sort best-first
        order = np.argsort(f_vals)
        X_sorted = X[order]
        Y_sorted = Y[order]
        Z_sorted = Z[order]
        f_sorted = f_vals[order]

        # Distribution updates
        yw = self._update_mean(Y_sorted, xp) # Eq. 42
        ps_norm = self._update_step_size(yw, Z_sorted, xp) # Eq. 43-44
        hsig = self._hsig(ps_norm, xp)
        self._update_covariance(yw, Y_sorted, hsig, xp) # Eq. 45-47
        self._update_eigensystem(function.n_calls, xp) # Appendix B.2

        # Write results back to space
        for i, ag in enumerate(space.agents):
            k = i % self.lam
            ag.position[:] = X_sorted[k].reshape(ag.position.shape)
            ag.fit = float(f_sorted[k])

            if ag.fit < space.best_agent.fit:
                space.best_agent.position = ag.position.copy()
                space.best_agent.fit = ag.fit
                space.best_agent.ts = int(time.time())


class CMAESTensor(Optimizer, _CMAESMixin):
    """
    CMA-ES —  Tensorized.
    """

    def __init__(
        self,
        params: Dict = None,
        sigma0: float = 0.3,
        **kwds,
    ):
        logger.info("Overriding class: Optimizer -> CMA-ES (Tensorized).")
        super().__init__()

        self.sigma0 = sigma0
        self.build(params)

        self.n: int = None
        self.lam: int = None
        self.mu: int = None
        self.weights = None 
        self.mueff: float = None

        # Step-size control
        self.cs: float = None
        self.ds: float = None

        # Covariance-matrix learning rates
        self.cc: float = None
        self.c1: float = None
        self.cmu: float = None

        self.m = None
        self.sigma: float = None
        self.C = None
        self.B = None
        self.D = None
        self.ps = None
        self.pc = None
        self.chiN: float = None
        self.g: int = None
        self.eigeneval: int = None

        # Cached bounds (GPU)
        self._lb = None
        self._ub = None

        self.best_position = None
        self.best_fit = None

        self.position = None
        self.fit = None
        
        self.DTYPE = None

        logger.info("Class overrided.")

    # ------------------------------------------------------------------

    def compile(self, space: _SingleObjectiveTensorSpace, **kwargs):
        xp = space.env.xp
        self.DTYPE = space.env.dtype
        n = len(space.lb)
        self.n = n
        self.lam = space.n_agents
        

        # Scalar hyper-parameters
        self._compute_hyperparams(n, xp)

        
        self.m = xp.mean(space.X, axis=0) 
        self.sigma = float(self.sigma0)
        self.g = 0
        self.eigeneval = 0


        # Evolution paths + covariance matrix
        self._init_state_arrays(n, xp)

        # Cast state arrays to env dtype
        self.C = self.C.astype(self.DTYPE)
        self.B = self.B.astype(self.DTYPE)
        self.D = self.D.astype(self.DTYPE)
        self.ps = self.ps.astype(self.DTYPE)
        self.pc = self.pc.astype(self.DTYPE)

    
        self._lb = space.lb
        self._ub = space.ub

        self.best_position = xp.zeros(n, dtype=self.DTYPE)
        self.best_fit = xp.asarray(xp.inf, dtype=self.DTYPE)


        self.position = xp.zeros((self.lam, self.n), dtype=self.DTYPE)
        self.fit = xp.full(self.lam, xp.inf, dtype=xp.float32)

        # Dummy operations for Kernel / CuPy JIT Warmup
        _w = xp.eye(2, dtype=self.DTYPE)
        xp.linalg.eigh(_w)                                      
        xp.random.randn(2, 2).astype(self.DTYPE)               
        (_w @ _w)
        xp.diag(xp.diag(_w))  
        xp.clip(_w, 0.0, 1.0)                                 
        xp.argsort(xp.array([1.0, 0.0], dtype=self.DTYPE))
        del _w

        logger.debug(
            "CMA-ES (Tensorized) compiled: n=%d  lambda=%d  µ=%d  µ_eff=%.2f  "
            "c1=%.4f  cµ=%.4f  cc=%.4f  cs=%.4f  ds=%.4f",
            n, self.lam, self.mu, self.mueff,
            self.c1, self.cmu, self.cc, self.cs, self.ds,
        )

    # ------------------------------------------------------------------

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function):
        """
        Initial evaluation — called once before the first update.
        """
        xp = space.env.xp

        space.F[:] = function(space.X, xp=xp)


        best_idx = xp.argmin(space.F)
        best_val = space.F[best_idx]
        improved = best_val < self.best_fit
        self.best_position = xp.where(improved, space.X[best_idx], self.best_position)
        self.best_fit = xp.where(improved, best_val, self.best_fit)

        space.best_agent.fit = float(self.best_fit)
        space.best_agent.ts = int(time.time())

        
        self.evaluate = lambda: None

    # ------------------------------------------------------------------

    def update(self, space: _SingleObjectiveTensorSpace, function: Function):
        """
        One CMA-ES generation — fully on GPU.
        """
        xp = space.env.xp
        self.g += 1

        # (Eq. 38-40)
        X, Y, Z = self._sample_population(xp)

        
        
        # Uses cached GPU bounds to avoid implicit lists casting
        X = self._clip_to_bounds(X, self._lb, self._ub, xp)

        f_vals = function(X, xp=xp)

        order = xp.argsort(f_vals)
        X_sorted = X[order]
        Y_sorted = Y[order]
        Z_sorted = Z[order]
        f_sorted = f_vals[order]

        yw = self._update_mean(Y_sorted, xp) # Eq. 42
        ps_norm = self._update_step_size(yw, Z_sorted, xp) # Eq. 43-44
        hsig = self._hsig(ps_norm, xp)
        self._update_covariance(yw, Y_sorted, hsig, xp) # Eq. 45-47
        self._update_eigensystem(function.n_calls, xp) # Appendix B.2

        improved = f_sorted[0] < self.best_fit
        self.best_position = xp.where(improved, X_sorted[0], self.best_position)
        self.best_fit = xp.where(improved, f_sorted[0], self.best_fit)
        
        space.X[:] = X_sorted
        space.F[:] = f_sorted
       
        space.best_agent.fit = float(self.best_fit)
        space.best_agent.ts = int(time.time())

    # ------------------------------------------------------------------

    def sync(self, space: _SingleObjectiveTensorSpace) -> None:
        """
        Updates the agents in the space with the current generation's positions and fitnesses.
        """

        xp = space.env.xp
        if xp.__name__ == 'cupy':
            for i, agent in enumerate(space.agents):
                agent.position[:] = xp.array(space.X[i]).reshape(-1,1).get()
                agent.fit = space.F[i].get()

            space.best_agent.position[:] = self.best_position.reshape(-1, 1).get()
            space.best_agent.fit = self.best_fit.get()
        else:
            for i, agent in enumerate(space.agents):
                agent.position[:] = xp.array(space.X[i]).reshape(-1, 1)
                agent.fit = space.F[i]

            space.best_agent.position[:] = self.best_position.reshape(-1, 1)
            space.best_agent.fit = self.best_fit

