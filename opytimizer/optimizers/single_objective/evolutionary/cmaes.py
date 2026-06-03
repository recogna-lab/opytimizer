"""
CMA-ES — Covariance Matrix Adaptation Evolution Strategy
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Environment, Function, Optimizer
from opytimizer.core.environment import Backend
from opytimizer.core.space import _SingleObjectiveSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------

class CMAES:
    """
        Reference
        ---------
        Hansen, N. (2023). The CMA Evolution Strategy: A Tutorial.
        arXiv:1604.00772v2 [cs.LG]. https://arxiv.org/abs/1604.00772
    """

    _registry: Dict[str, type] = {}

    def __new__(cls, env: Optional[Environment] = None, **kwargs):
        if env is None:
            env = Environment().set_backend("cpu")
        if cls is CMAES:
            target = cls._registry.get(env.backend)
            return super().__new__(target)
        return super().__new__(cls)

    def __init_subclass__(cls, backend: Backend = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if backend:
            key = backend.value if hasattr(backend, "value") else backend
            CMAES._registry[key] = cls


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
        """
        """

        # Population / parent sizes  (Table 1 / Appendix B)
        self.lam = 4 + int(np.floor(3 * np.log(n)))
        self.mu = self.lam // 2

        # Raw log weights  (Eq. 49 / 53)
        w_raw   = np.log(self.lam / 2.0 + 0.5) - np.log(np.arange(1, self.lam + 1))
        w_pos   = w_raw[: self.mu]
        w_pos_n = w_pos / w_pos.sum() # positive weights sum to 1

        # Effective variance selection mass  (Eq. 8)
        self.mueff = float(1.0 / np.sum(w_pos_n ** 2))

        w_neg_raw = w_raw[self.mu:]
        mueff_neg = (
            w_neg_raw.sum() ** 2 / np.sum(w_neg_raw ** 2)
            if len(w_neg_raw) else 1.0
        )

        # Step-size control constants  (Eq. 55)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.ds = (
            1.0
            + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0)
            + self.cs
        )

        # Covariance-matrix learning rates  (Eq. 56-58)
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        alpha_cov = 2.0
        self.cmu = min(
            1.0 - self.c1,
            alpha_cov
            * (self.mueff - 2.0 + 1.0 / self.mueff)
            / ((n + 2.0) ** 2 + alpha_cov * self.mueff / 2.0),
        )

        # Negative-weight scaling  (Eq. 50-52)
        alpha_mu = 1.0 + self.c1 / self.cmu  if self.cmu > 0 else 1.0
        alpha_mueff = 1.0 + 2.0 * mueff_neg / (self.mueff + 2.0)
        alpha_posdef = (
            (1.0 - self.c1 - self.cmu) / (n * self.cmu) if self.cmu > 0 else 1.0
        )
        alpha_neg = min(alpha_mu, alpha_mueff, alpha_posdef)

        # Final weight vector  (Eq. 53)
        w_neg_sum = np.abs(w_neg_raw).sum() if len(w_neg_raw) else 1.0
        w_neg_sc = alpha_neg / w_neg_sum * w_neg_raw
        self.weights = np.concatenate([w_pos_n, w_neg_sc])  

        # E||N(0,I)||  (Appendix A)
        self.chiN = n ** 0.5 * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

    # Sampling  (Eq. 38-40)
    def _sample_population(self, xp) -> tuple:
        """
        """
        Z = xp.random.randn(self.lam, self.n)  # Eq. 38
        BD = self.B @ self.D  # Eq. 39 (factor)
        Y = Z @ BD.T #  Eq. 39
        X = self.m + self.sigma * Y  #  Eq. 40
        return X, Y, Z

    # Bounds repair (Appendix B.5)
    def _clip_to_bounds(self, X, lb, ub, xp):
        return xp.clip(X, lb, ub)

   
    # Mean update  (Eq. 42)
    def _update_mean(self, Y_sorted, xp):
        """
        Shift the distribution mean toward the weighted-recombination step.

        Parameters
        ----------
        Y_sorted : (lam, n)  steps sorted best-first

        Returns
        -------
        yw : (n,)  weighted mean step  <y>_w  (used in path updates)
        """
        w  = xp.asarray(self.weights[: self.mu]) # (mu,)
        yw = w @ Y_sorted[: self.mu] # (n,)  Eq. 42

        self.m = self.m + self.sigma * yw
        return yw

    
    # Step-size control  (Eq. 43-44)
    def _update_step_size(self, yw, Z_sorted, xp):
        """
        Update the cumulative step-size path ps and adapt σ.

        Parameters
        ----------
        yw       : (n,)   weighted mean step in *y*-space
        Z_sorted : (lam, n)  standard-normal noise sorted best-first
        """
        # Weighted mean in isotropic z-space  (Appendix B.1)
        w  = xp.asarray(self.weights[: self.mu])
        zw = w @ Z_sorted[: self.mu] # (n,)

        # (Appendix B.1)
        invsqrtC_yw = self.B @ zw

        # Cumulative step-size path  (Eq. 43)
        self.ps = (
            (1.0 - self.cs) * self.ps
            + float(np.sqrt(self.cs * (2.0 - self.cs) * self.mueff))
            * invsqrtC_yw
        )

        # Step-size update  (Eq. 44)
        ps_norm = float(xp.linalg.norm(self.ps))
        self.sigma *= float(np.exp((self.cs / self.ds) * (ps_norm / self.chiN - 1.0)))

    
    # Stall indicator  (Appendix A / Fig. 6)
    def _hsig(self, xp) -> int:
        """Return 1 when ps indicates the path has not stalled."""
        ps_norm = float(xp.linalg.norm(self.ps))
        correction = float(np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.g + 1))))
        threshold = (1.4 + 2.0 / (self.n + 1.0)) * self.chiN
        return int((ps_norm / correction) < threshold)

    # Covariance matrix update  (Eq. 45-47)
    def _update_covariance(self, yw, Y_sorted, hsig: int, xp):
        """
        Rank-1 + rank-μ update of the covariance matrix.

        Parameters
        ----------
        yw       : (n,)    weighted mean step
        Y_sorted : (lam, n) steps sorted best-first
        hsig     : int     stall indicator (0 or 1)
        """

        # Cumulative evolution path  (Eq. 45)
        self.pc = (
            (1.0 - self.cc) * self.pc
            + hsig
            * float(np.sqrt(self.cc * (2.0 - self.cc) * self.mueff))
            * yw
        )

        # Mahalanobis distance for negative-weight scaling  (Eq. 46)
        BtY = Y_sorted @ self.B  # (lam, n)
        d_diag = xp.diag(self.D) # (n,)
        DinvBtY = BtY / d_diag # (lam, n)
        mahal_sq = xp.sum(DinvBtY ** 2, axis=1)  # (lam,)

        w_all   = xp.asarray(self.weights) # (lam,)
        w_circle= xp.where(
            w_all >= 0,
            w_all,
            w_all * self.n / (mahal_sq + 1e-20), # Eq. 46
        )

        # Decay factor  (Eq. 47)
        delta_hsig = (1.0 - hsig) * self.cc * (2.0 - self.cc)
        decay = (
            1.0
            + self.c1 * delta_hsig
            - self.c1
            - self.cmu * float(xp.sum(xp.asarray(self.weights)))
        )

        # Rank-1 component  (Eq. 45)
        rank1 = xp.outer(self.pc, self.pc)

        # Rank-μ component  (Eq. 47)
        rankmu = (Y_sorted * w_circle[:, None]).T @ Y_sorted

        self.C = decay * self.C + self.c1 * rank1 + self.cmu * rankmu

        # Enforce symmetry
        self.C = xp.triu(self.C) + xp.triu(self.C, 1).T

   
    # Eigendecomposition (lazy)  (Appendix B.2)
    def _update_eigensystem(self, total_evals: int, xp):
        """
        Recompute B, D from C at most once every 1/(10 n (c1+cμ)) evaluations.
        """
        c_sum = self.c1 + self.cmu
        interval = max(1, int(np.floor(1.0 / (10.0 * self.n * c_sum))))

        if self.g - self.eigeneval >= interval:
            self.eigeneval = self.g

            eigenvalues, self.B = xp.linalg.eigh(self.C)  # ascending order
            eigenvalues = xp.maximum(eigenvalues, 1e-20) # numerical guard
            self.D = xp.diag(xp.sqrt(eigenvalues))


# CPU backend
@dataclass
class _CMAESDefault(Optimizer, _CMAESMixin, CMAES, backend=Backend.CPU):
    """CMA-ES — CPU / numpy backend."""

    def __init__(
        self,
        params: Dict = None,
        sigma0: float = 0.3,
        **kwds,
    ):
        logger.info("Overriding class: Optimizer -> CMA-ES (CPU).")
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

        logger.info("Class overrided.")

    # ------------------------------------------------------------------

    def compile(self, space: _SingleObjectiveSpace, **kwargs):
        xp = np  # CPU backend always uses numpy
        n = len(space.lb)
        self.n = n

        self._compute_hyperparams(n, xp)

        # Initialize mean from current population  (Fig. 6)
        self.m = np.mean(
            [ag.position.flatten() for ag in space.agents], axis=0
        )
        self.sigma = self.sigma0
        self.g = 0
        self.eigeneval = 0

        self._init_state_arrays(n, xp)

        logger.debug(
            "CMA-ES (CPU) compiled: n=%d  lambda=%d  µ=%d  µ_eff=%.2f  "
            "c1=%.4f  cµ=%.4f  cc=%.4f  cs=%.4f  ds=%.4f",
            n, self.lam, self.mu, self.mueff,
            self.c1, self.cmu, self.cc, self.cs, self.ds,
        )

    # ------------------------------------------------------------------

    def update(self, space: _SingleObjectiveSpace, function: Function):
        xp = np
        self.g += 1

        lb = np.array(space.lb).flatten()
        ub = np.array(space.ub).flatten()

        # Sample population  (Eq. 38-40)
        X, Y, Z = self._sample_population(xp)
        X = self._clip_to_bounds(X, lb, ub, xp)

        # Evaluate — one call per candidate (CPU path)
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
        self._update_step_size(yw, Z_sorted, xp) # Eq. 43-44
        hsig = self._hsig(xp)
        self._update_covariance(yw, Y_sorted, hsig, xp) # Eq. 45-47
        self._update_eigensystem(function.n_calls, xp) # Appendix B.2

        # Write results back to space
        for i, ag in enumerate(space.agents):
            k = i % self.lam
            ag.position = X_sorted[k].reshape(ag.position.shape)
            ag.fit = float(f_sorted[k])

            if ag.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(ag.position)
                space.best_agent.fit = copy.deepcopy(ag.fit)
                space.best_agent.ts = int(time.time())

    # ------------------------------------------------------------------

    def evaluate(self, space: _SingleObjectiveSpace, function: Function):
        if function.n_calls == 0:
            for agent in space.agents:
                agent.fit = function(agent.position)

                if agent.fit < space.best_agent.fit:
                    space.best_agent.position = copy.deepcopy(agent.position)
                    space.best_agent.fit = copy.deepcopy(agent.fit)
                    space.best_agent.ts = int(time.time())


# CUDA backend
@dataclass
class _CMAESCuda(Optimizer, _CMAESMixin, CMAES, backend=Backend.CUDA):
    """
    CMA-ES — CUDA / CuPy backend.
    """

    def __init__(
        self,
        params: Dict = None,
        sigma0: float = 0.3,
        **kwds,
    ):
        logger.info("Overriding class: Optimizer -> CMA-ES (CUDA).")
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

        # Distribution state  (GPU tensors, allocated in compile)
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

        # dtype mirror from environment
        self.DTYPE = None

        self._isFirstIteration = True

        logger.info("Class overrided.")

    # ------------------------------------------------------------------

    def compile(self, space: _SingleObjectiveSpace, **kwargs):
        xp = space.env.xp
        self.DTYPE = space.env.dtype
        n = len(space.lb)
        self.n = n

        # Scalar hyper-parameters
        self._compute_hyperparams(n, np)

        # Initialize mean from current population  (Fig. 6)
        positions = xp.stack(
            [xp.asarray(ag.position.flatten(), dtype=self.DTYPE)
             for ag in space.agents]
        )
        self.m = xp.mean(positions, axis=0) # (n,)  on GPU
        self.sigma = float(self.sigma0)
        self.g = 0
        self.eigeneval = 0

        # Evolution paths + covariance matrix
        self._init_state_arrays(n, xp)

        # Cast state arrays to env dtype
        self.C = self.C.astype(xp.float64)
        self.B = self.B.astype(xp.float64)
        self.D = self.D.astype(xp.float64)
        self.ps = self.ps.astype(xp.float64)
        self.pc = self.pc.astype(xp.float64)

        # Cache bounds as GPU vectors 
        self._lb = xp.asarray(space.lb, dtype=self.DTYPE).flatten()
        self._ub = xp.asarray(space.ub, dtype=self.DTYPE).flatten()

        _w = xp.eye(2, dtype=self.DTYPE)
        xp.linalg.eigh(_w)                                      
        xp.random.randn(2, 2).astype(self.DTYPE)               
        (_w @ _w)
        xp.diag(xp.diag(_w))  
        xp.clip(_w, 0.0, 1.0)                                 
        xp.argsort(xp.array([1.0, 0.0], dtype=self.DTYPE))
        del _w


        logger.debug(
            "CMA-ES (CUDA) compiled: n=%d  lambda=%d  µ=%d  µ_eff=%.2f  "
            "c1=%.4f  cµ=%.4f  cc=%.4f  cs=%.4f  ds=%.4f",
            n, self.lam, self.mu, self.mueff,
            self.c1, self.cmu, self.cc, self.cs, self.ds,
        )

        

    # ------------------------------------------------------------------

    
    def update(self, space: _SingleObjectiveSpace, function: Function):
        """
        One CMA-ES generation — fully on GPU.
        """
        xp = space.env.xp
        self.g += 1

        # (Eq. 38-40)
        X, Y, Z = self._sample_population(xp) 
        X = self._clip_to_bounds(X, self._lb, self._ub, xp)

       
        
        f_vals = function(X, xp=xp)
       
        order = xp.argsort(f_vals)
        X_sorted = X[order]
        Y_sorted = Y[order]
        Z_sorted = Z[order]
        f_sorted = f_vals[order]

        
        yw = self._update_mean(Y_sorted, xp) # Eq. 42
        self._update_step_size(yw, Z_sorted, xp) # Eq. 43-44
        hsig = self._hsig(xp)
        self._update_covariance(yw, Y_sorted, hsig, xp) # Eq. 45-47
        self._update_eigensystem(function.n_calls, xp) # Appendix B.2

        #  We pull only the scalars/positions we actually need.
        agent_shape = space.agents[0].position.shape

        
       
        f_cpu = xp.asnumpy(f_sorted).ravel().tolist()
        xp.cuda.Stream.null.synchronize()

        for i, ag in enumerate(space.agents):
            k = i % self.lam
            gpu_pos = xp.copy(X_sorted[k]).reshape(agent_shape).astype(self.DTYPE)
            ag.position[:] = gpu_pos
            ag.fit = f_cpu[k]

            if ag.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(ag.position)
                space.best_agent.fit = float(ag.fit)
                space.best_agent.ts = int(time.time())

        

    # ------------------------------------------------------------------

    def evaluate(self, space: _SingleObjectiveSpace, function: Function):
        """
        Initial evaluation — called once before the first update.
        """
        xp = space.env.xp
        
        if self._isFirstIteration:
            self._isFirstIteration = False
            # Build (n_agents, n) 
            P = xp.stack(
                [xp.asarray(ag.position.flatten(), dtype=self.DTYPE)
                 for ag in space.agents]
            )                                           

            f_vals = function(P, xp=xp)
            
            f_cpu  = xp.asnumpy(f_vals)
            for i, ag in enumerate(space.agents):
                ag.fit = float(f_cpu[i])

                if ag.fit < space.best_agent.fit:
                    space.best_agent.position = copy.deepcopy(ag.position)
                    space.best_agent.fit = copy.deepcopy(ag.fit)
                    space.best_agent.ts = int(time.time())

        xp.cuda.Stream.null.synchronize()
        