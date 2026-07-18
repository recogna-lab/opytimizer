"""L-SHADE"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import copy
import time 
from typing import List, Any, Dict
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer, Function
from opytimizer.core.space import _SingleObjectiveSpace
from opytimizer.utils import logging
from opytimizer.core.environment import Backend
from scipy.stats import cauchy
logger = logging.get_logger(__name__)


class LSHADE(Optimizer):
    def __init__(self,
                 params: Dict = None,
                 MAX_NFE: int = 100,
                 H: int = 100,
                 p: float = 0.11,
                 f_arc: float = 2.6,
                 **kwds):
        
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
        """"""
       
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

       
        idx_r1 = np.random.randint(0, N_G, size=N_G)
        collision_r1 = (idx_r1 == np.arange(N_G))
        while np.any(collision_r1):
            idx_r1[collision_r1] = np.random.randint(0, N_G, size=np.sum(collision_r1))
            collision_r1 = (idx_r1 == np.arange(N_G))
        X_r1 = X[idx_r1]

        
        if len(self.A) > 0:
            A_matrix = np.array(self.A)
            X_union = np.vstack([X, A_matrix])
        else:
            X_union = X
        n_union = len(X_union)

        
        idx_r2 = np.random.randint(0, n_union, size=N_G)
        collision_r2 = (idx_r2 == np.arange(N_G)) | (idx_r2 == idx_r1)
        while np.any(collision_r2):
            idx_r2[collision_r2] = np.random.randint(0, n_union, size=np.sum(collision_r2))
            collision_r2 = (idx_r2 == np.arange(N_G)) | (idx_r2 == idx_r1)
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

    def _crossover(self, CR: np.ndarray, V: np.ndarray, space: _SingleObjectiveSpace) -> np.ndarray:
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
        invalid_mask = (F <= 0.0)
        while np.any(invalid_mask):
            num_invalid = np.sum(invalid_mask)
            new_samples = cauchy.rvs(loc=mu_F[invalid_mask], scale=0.1, size=num_invalid)
            F[invalid_mask] = new_samples
            invalid_mask = (F <= 0.0)
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

            indices_to_keep = np.random.choice(len(self.A), size=self.N_A, replace=False)
            self.A = [self.A[idx] for idx in indices_to_keep]

        if len(S_CR) > 0:
            weights = DELTA_F / np.sum(DELTA_F)
            
            if np.isnan(self.M_CR[self.k]) or np.max(S_CR) == 0:
                self.M_CR[self.k] = np.nan
            else:
                self.M_CR[self.k] = np.sum(weights * (S_CR ** 2)) / np.sum(weights * S_CR)
                
            self.M_F[self.k] = np.sum(weights * (S_F ** 2)) / np.sum(weights * S_F)
            
            self.k = (self.k + 1) % self.H

        N_min = 4
        N_target = int(np.round(((N_min - self.N_init) / self.MAX_NFE) * function.n_calls + self.N_init))
        N_target = max(N_min, N_target)

        if N_target < self.N_G:
            space.agents.sort(key=lambda ag: ag.fit)
            space.agents = space.agents[:N_target]
            
            self.N_G = N_target
            self.N_A = int(np.round(self.f_arc * self.N_G))
            
            if len(self.A) > self.N_A:
                self.A = self.A[:self.N_A]


    def evaluate(self, space, function):
        if function.n_calls == 0:
            for agent in space.agents:
                agent.fit = function(agent.position)
                
                if agent.fit < space.best_agent.fit:
                    space.best_agent.position = copy.deepcopy(agent.position)
                    space.best_agent.fit = copy.deepcopy(agent.fit)
                    space.best_agent.ts = int(time.time())

class LSHADECuda(Optimizer):
    def __init__(self,
                 params: Dict = None,
                 MAX_NFE: int = 100,
                 H: int = 100,
                 p: float = 0.11,
                 f_arc: float = 2.6,
                 **kwds):
        
        logger.info("Overriding class: Optimizer -> L-SHADE (CUDA).")
        
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
        self.DTYPE = space.env.xp.float32
        self.N_G = space.n_agents
        self.N_init = space.n_agents
        self.M_CR = np.full(self.H, 0.5)
        self.M_F = np.full(self.H, 0.5)
        self.k = 0  
        self.N_A = int(np.round(self.f_arc * self.N_G))
        self.A = []

    def _mutate_hybrid(self, X_gpu: Any, F_cpu: np.ndarray, space: _SingleObjectiveSpace, xp: Any) -> Any:
        N_G = self.N_G

       
        fits_cpu = np.array([float(ag.fit) for ag in space.agents])
        
      
        pbest_bound = max(2, int(np.round(N_G * self.p)))
        p_best_indices_pool = np.argsort(fits_cpu)[:pbest_bound]
        
        idx_pbest_cpu = np.random.choice(p_best_indices_pool, size=N_G)

        idx_r1_cpu = np.random.randint(0, N_G - 1, size=N_G)
        idx_r1_cpu = np.where(idx_r1_cpu >= np.arange(N_G), idx_r1_cpu + 1, idx_r1_cpu)

        n_union = N_G + len(self.A)

        idx_r2_cpu = np.random.randint(0, n_union - 2, size=N_G)
        exc1 = np.minimum(np.arange(N_G), idx_r1_cpu)
        exc2 = np.maximum(np.arange(N_G), idx_r1_cpu)
        idx_r2_cpu = np.where(idx_r2_cpu >= exc1, idx_r2_cpu + 1, idx_r2_cpu)
        idx_r2_cpu = np.where(idx_r2_cpu >= exc2, idx_r2_cpu + 1, idx_r2_cpu)

       
        idx_pbest_gpu = xp.asarray(idx_pbest_cpu)
        idx_r1_gpu = xp.asarray(idx_r1_cpu)
        idx_r2_gpu = xp.asarray(idx_r2_cpu)
        F_gpu = xp.asarray(F_cpu, dtype=self.DTYPE)[:, xp.newaxis]

        
        X_pbest_gpu = X_gpu[idx_pbest_gpu]
        X_r1_gpu = X_gpu[idx_r1_gpu]

        if len(self.A) > 0:
           
            A_gpu = xp.asarray(self.A, dtype=self.DTYPE)
            X_union_gpu = xp.vstack([X_gpu, A_gpu])
        else:
            X_union_gpu = X_gpu

        X_r2_gpu = X_union_gpu[idx_r2_gpu]

        
        V_gpu = X_gpu + F_gpu * (X_pbest_gpu - X_gpu) + F_gpu * (X_r1_gpu - X_r2_gpu)

        
        lb = space.lb.flatten()
        ub = space.ub.flatten()

        V_gpu = xp.where(V_gpu < lb, (lb + X_gpu) / 2.0, V_gpu)
        V_gpu = xp.where(V_gpu > ub, (ub + X_gpu) / 2.0, V_gpu)

        return V_gpu

    def _crossover_gpu(self, CR_cpu: np.ndarray, V_gpu: Any, X_gpu: Any, space: _SingleObjectiveSpace, xp: Any) -> Any:
        N_G, D = X_gpu.shape
        CR_gpu = xp.asarray(CR_cpu, dtype=self.DTYPE)[:, xp.newaxis]

        rand_matrix = xp.random.rand(N_G, D, dtype=self.DTYPE)
        crossover_mask = rand_matrix <= CR_gpu

        j_rand_gpu = xp.random.randint(0, D, size=N_G)
        crossover_mask[xp.arange(N_G), j_rand_gpu] = True

        U_gpu = xp.where(crossover_mask, V_gpu, X_gpu)
        return U_gpu

    def update(self, space: _SingleObjectiveSpace, function: Function):
        xp = space.env.xp  

        
        X_gpu = xp.stack([ag.position.flatten() for ag in space.agents])

        
        selected_indices = np.random.randint(low=0, high=self.H, size=self.N_G)
        mu_CR = self.M_CR[selected_indices]
        mu_F = self.M_F[selected_indices]

        CR = np.random.normal(loc=mu_CR, scale=0.1)
        CR = np.where(np.isnan(mu_CR), 0.0, CR)
        CR = np.clip(CR, 0.0, 1.0)

        F = cauchy.rvs(loc=mu_F, scale=0.1, size=self.N_G)
        invalid_mask = (F <= 0.0)
        while np.any(invalid_mask):
            num_invalid = np.sum(invalid_mask)
            F[invalid_mask] = cauchy.rvs(loc=mu_F[invalid_mask], scale=0.1, size=num_invalid)
            invalid_mask = (F <= 0.0)
        F = np.clip(F, 0.0, 1.0)

        V_gpu = self._mutate_hybrid(X_gpu, F, space, xp)
        U_gpu = self._crossover_gpu(CR, V_gpu, X_gpu, space, xp)   

        
        f_U_gpu = function(U_gpu, xp=xp)
        
       
        f_U_cpu = xp.asnumpy(f_U_gpu).squeeze()
        X_old_cpu = xp.asnumpy(X_gpu) 
        
        
        fits_cpu = np.array([float(ag.fit) for ag in space.agents])
        
        survival_mask = f_U_cpu <= fits_cpu
        success_mask = f_U_cpu < fits_cpu  

        S_CR = CR[success_mask]
        S_F = F[success_mask]
        DELTA_F = np.abs(fits_cpu[success_mask] - f_U_cpu[success_mask])

        
        for i, ag in enumerate(space.agents):
            if survival_mask[i]:
                ag.position = U_gpu[i].reshape(ag.position.shape).copy()
                
                ag.fit = xp.asarray(f_U_cpu[i], dtype=self.DTYPE) 
                
                if success_mask[i]:
                    self.A.append(X_old_cpu[i])
                
                if f_U_cpu[i] < float(space.best_agent.fit):
                    space.best_agent.position = ag.position.copy()
                    space.best_agent.fit = ag.fit.copy()
                    space.best_agent.ts = int(time.time())

        
        if len(self.A) > self.N_A:
            indices_to_keep = np.random.choice(len(self.A), size=self.N_A, replace=False)
            self.A = [self.A[idx] for idx in indices_to_keep]

        
        if len(S_CR) > 0:
            weights = DELTA_F / np.sum(DELTA_F)
            
            if np.isnan(self.M_CR[self.k]) or np.max(S_CR) == 0:
                self.M_CR[self.k] = np.nan
            else:
                self.M_CR[self.k] = np.sum(weights * (S_CR ** 2)) / np.sum(weights * S_CR)
                
            self.M_F[self.k] = np.sum(weights * (S_F ** 2)) / np.sum(weights * S_F)
            self.k = (self.k + 1) % self.H

       
        N_min = 4
        N_target = int(np.round(((N_min - self.N_init) / self.MAX_NFE) * function.n_calls + self.N_init))
        N_target = max(N_min, N_target)

        if N_target < self.N_G:
            space.agents.sort(key=lambda ag: float(ag.fit))
            space.agents = space.agents[:N_target]
            
            self.N_G = N_target
            self.N_A = int(np.round(self.f_arc * self.N_G))
            if len(self.A) > self.N_A:
                self.A = self.A[:self.N_A]

    def evaluate(self, space, function):
        xp = space.env.xp
        
        if function.n_calls == 0:
            X_gpu = xp.stack([ag.position.flatten() for ag in space.agents])
            fits_gpu = function(X_gpu, xp=xp)
            fits_cpu = xp.asnumpy(fits_gpu).squeeze()

            for i, agent in enumerate(space.agents):
                
                agent.fit = xp.asarray(fits_cpu[i], dtype=self.DTYPE)
                
                if fits_cpu[i] < float(space.best_agent.fit):
                    space.best_agent.position = agent.position.copy()
                    space.best_agent.fit = agent.fit.copy()
                    space.best_agent.ts = int(time.time())