"""
CMA-ES — Covariance Matrix Adaptation Evolution Strategy
All equation numbers in comments refer to that paper.
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



@dataclass
class _CMAESDefault(Optimizer, CMAES, backend=Backend.CPU):
    """
    """

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


        # Dimension of the search space
        self.n: int = None

        # Population / parent sizes 
        self.lam: int = None 
        self.mu: int = None  

        # w_i  (Eq. 49 / 53 / Table 1)
        self.weights: np.ndarray = None

        # (Eq. 8)
        self.mueff: float = None

       

        # Step-size control (CSA)
        self.cs: float = None # (Eq. 55)
        self.ds: float = None # (Eq. 55)

        # Covariance matrix adaptation
        self.cc: float = None # (Eq. 56)
        self.c1: float = None # (Eq. 57)
        self.cmu: float = None # (Eq. 58)


        # Distribution mean
        self.m: np.ndarray = None

        # Overall step-size (scalar)
        self.sigma: float = None

        # Covariance matrix
        self.C: np.ndarray = None

      
        self.B: np.ndarray = None
        self.D: np.ndarray = None

        # Evolution path for step-size control (Eq. 31 / 43)
        # Accumulates (conjugate) successive steps to detect correlations.
        self.ps: np.ndarray = None

        # Evolution path for covariance matrix  (Eq. 24 / 45)
        # Reintroduces sign information lost in the outer-product update
        self.pc: np.ndarray = None

        # Expected Euclidean norm of N(0,I)  E||N(0,I)||  (Appendix A)
        self.chiN: float = None

        # Generation counter  g 
        self.g: int = None

        # How often eigendecomposition has been re-computed
        # (lazy update every - Appendix B.2)
        self.eigeneval: int = None

        logger.info("Class overrided.")


    def compile(self, space: _SingleObjectiveSpace, **kwargs):
        """
        """

        n = len(space.lb)  # problem dimension
        self.n = n

        # population / parent sizes  (Eq. 48 / Table 1)
        # local convergence (Appendix B).
        self.lam = 4 + int(np.floor(3 * np.log(n)))
        self.mu = self.lam // 2  
        # (Eq. 49 / 53 / Table 1)
        w_raw = np.log(self.lam / 2.0 + 0.5) - np.log(np.arange(1, self.lam + 1))

        # (Eq. 8)
        w_pos = w_raw[: self.mu]
        w_pos_norm = w_pos / w_pos.sum()          # positive weights sum to 1
        self.mueff = 1.0 / np.sum(w_pos_norm ** 2)

        w_neg_raw = w_raw[self.mu :]
        mueff_neg = w_neg_raw.sum() ** 2 / np.sum(w_neg_raw ** 2) if len(w_neg_raw) else 1.0


        # (Eq. 55)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.ds = (
            1.0
            + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0)
            + self.cs
        )

        # (Eq. 56)
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)

        # (Eq. 57)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)

        # (Eq. 58)
        alpha_cov = 2.0
        self.cmu = min(
            1.0 - self.c1,
            alpha_cov
            * (self.mueff - 2.0 + 1.0 / self.mueff)
            / ((n + 2.0) ** 2 + alpha_cov * self.mueff / 2.0),
        )

        # (Eq. 50)
        alpha_mu = 1.0 + self.c1 / self.cmu if self.cmu > 0 else 1.0
        # (Eq. 51)
        alpha_mueff = 1.0 + 2.0 * mueff_neg / (self.mueff + 2.0)
        # (Eq. 52)
        alpha_posdef = (1.0 - self.c1 - self.cmu) / (n * self.cmu) if self.cmu > 0 else 1.0

        alpha_neg = min(alpha_mu, alpha_mueff, alpha_posdef)

        # Normalize weights  (Eq. 53)
        w_neg_raw_sum = np.abs(w_neg_raw).sum() if len(w_neg_raw) else 1.0
        w_neg_scaled = alpha_neg / w_neg_raw_sum * w_neg_raw  # still negative values

        self.weights = np.concatenate([w_pos_norm, w_neg_scaled]) 

        # (Appendix A) 
        self.chiN = n ** 0.5 * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # (Fig. 6 — Initialization)
        self.m = np.mean(
            [ag.position.flatten() for ag in space.agents], axis=0
        )

        self.sigma = self.sigma0

        # C = I  
        self.C = np.eye(n)

        # Eigendecomposition
        self.B = np.eye(n)
        self.D = np.eye(n) 

        # Evolution paths initialized to zero
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)

        self.g = 0
        self.eigeneval = 0

        logger.debug(
            "CMA-ES compiled: n=%d  sigma=%d  µ=%d  µ_eff=%.2f  "
            "c1=%.4f  cµ=%.4f  cc=%.4f  cs=%.4f  ds=%.4f",
            n, self.lam, self.mu, self.mueff,
            self.c1, self.cmu, self.cc, self.cs, self.ds,
        )

    #  Eq. 38-40

    def _sample_population(self) -> np.ndarray:
        """
        """

        Z = np.random.randn(self.n, self.lam).T  # Eq. 38

        BD = self.B @ self.D  
        Y = Z @ BD.T  # Eq. 39

        X = self.m + self.sigma * Y   # Eq. 40

        return X, Y, Z

    #  Appendix B.5 (simple repair)

    def _clip_to_bounds(
        self, X: np.ndarray, lb: np.ndarray, ub: np.ndarray
    ) -> np.ndarray:
        """
        """
        return np.clip(X, lb, ub)

    #  Eq. 9 / 42

    def _update_mean(self, Y_sorted: np.ndarray) -> np.ndarray:
        """
        """

        yw = self.weights[: self.mu] @ Y_sorted[: self.mu]  # shape (n,)

        # (Eq. 54 / Table 1)
        delta_m = self.sigma * yw                           

        self.m = self.m + delta_m
        return yw   # return normalised step  <y>_w  (used in path updates)

    # Eq. 31 / 43-44

    def _update_step_size(self, yw: np.ndarray, Z_sorted: np.ndarray):
        """
        """

        #  (Appendix B.1)
        zw = self.weights[: self.mu] @ Z_sorted[: self.mu]  
        invsqrtC_yw = self.B @ zw                           

        # (Eq. 43)
        self.ps = (
            (1.0 - self.cs) * self.ps
            + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * invsqrtC_yw
        )

        # (Eq. 44)
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp(
            (self.cs / self.ds) * (ps_norm / self.chiN - 1.0)
        )

    #  (Appendix A / Fig. 6)

    def _hsig(self) -> int:
        """
        """
        ps_norm = np.linalg.norm(self.ps)
        # Correction for finite path length at early generations
        correction = np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.g + 1)))
        threshold = (1.4 + 2.0 / (self.n + 1.0)) * self.chiN
        return int((ps_norm / correction) < threshold)
    

    #   Eq. 45-47


    def _update_covariance(
        self, yw: np.ndarray, Y_sorted: np.ndarray, hsig: int
    ):
        """
        """

       
        self.pc = (
            (1.0 - self.cc) * self.pc
            + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * yw
        )

       
        BtY = Y_sorted @ self.B 
        d_diag = np.diag(self.D) 
        DinvBtY = BtY / d_diag 
        mahal_sq = np.sum(DinvBtY ** 2, axis=1)

        w_circle = np.where(
            self.weights >= 0,
            self.weights,
            self.weights * self.n / (mahal_sq + 1e-20),   # Eq. 46
        )

        
        sum_w = np.sum(self.weights)  
        delta_hsig = (1.0 - hsig) * self.cc * (2.0 - self.cc)

        decay = 1.0 + self.c1 * delta_hsig - self.c1 - self.cmu * sum_w

        
        rank1 = np.outer(self.pc, self.pc) 

    
        rankmu = (Y_sorted * w_circle[:, None]).T @ Y_sorted  

        self.C = (
            decay * self.C
            + self.c1 * rank1
            + self.cmu * rankmu
        )

    
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

    #  Appendix B.2
   

    def _update_eigensystem(self, total_evals: int):
        """
        """
        c_sum = self.c1 + self.cmu
        update_interval = max(1, int(np.floor(1.0 / (10.0 * self.n * c_sum))))

        if total_evals - self.eigeneval >= update_interval:
            self.eigeneval = total_evals

            # Eigendecomposition:
            eigenvalues, self.B = np.linalg.eigh(self.C)   # eigenvalues sorted ascending
            # Clip for numerical safety (eigenvalues must be > 0)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.D = np.diag(np.sqrt(eigenvalues))          

   

    def update(self, space: _SingleObjectiveSpace, function: Function):
        """
        """
        self.g += 1

        lb = np.array(space.lb).flatten()  
        ub = np.array(space.ub).flatten()  

       
        # (Eq. 38-40)
       
        X, Y, Z = self._sample_population()    
        X = self._clip_to_bounds(X, lb, ub)


        f_vals = np.empty(self.lam)
        for k in range(self.lam):
           
            position = X[k].reshape(-1, 1)
            f_vals[k] = float(function(position))

     
        order = np.argsort(f_vals)        
        X_sorted = X[order]              
        Y_sorted = Y[order]             
        Z_sorted = Z[order]                
        f_sorted = f_vals[order]


        # (Eq. 42)  
      
        yw = self._update_mean(Y_sorted)    # shape (n,)


        #  (Eq. 43-44)
      
        self._update_step_size(yw, Z_sorted)

     
        hsig = self._hsig()


        #  (Eq. 45-47)

        self._update_covariance(yw, Y_sorted, hsig)

    
        #  (Appendix B.2)
  
        self._update_eigensystem(function.n_calls)


        for i, ag in enumerate(space.agents):
            
            k = i % self.lam
            ag.position = X_sorted[k].reshape(ag.position.shape)
            ag.fit = float(f_sorted[k])

            if ag.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(ag.position)
                space.best_agent.fit = copy.deepcopy(ag.fit)
                space.best_agent.ts = int(time.time())


    def evaluate(self, space: _SingleObjectiveSpace, function: Function):
        """
        """
        if function.n_calls == 0:
            for agent in space.agents:
                agent.fit = function(agent.position)

                if agent.fit < space.best_agent.fit:
                    space.best_agent.position = copy.deepcopy(agent.position)
                    space.best_agent.fit = copy.deepcopy(agent.fit)
                    space.best_agent.ts = int(time.time())