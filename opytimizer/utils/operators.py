import copy
from abc import ABC, abstractmethod

import numpy as np

from typing_extensions import Literal
from typing import Union, List, Tuple
import opytimizer.math.random as r
from opytimizer.core.agent import Agent
from opytimizer.core import Environment
from opytimizer.core.environment import Backend
import opytimizer.utils.exception as e

class BatchList(list):
    """Native python list carrying GPU tensors """
    def __init__(self, iterable, tensor=None):
        super().__init__(iterable)
        self.tensor = tensor
    


class BaseCrossover(ABC):
    """Abstract base class for crossover operators."""
    
    def __init__(self, rate: float = 1.0, return_mode: Literal['first','second', 'both', 'random'] = 'both') -> None:
        self.rate = rate
        self.return_mode = return_mode
        self.env = Environment().set_backend('cpu').set_dtype('float64')


             
    def _return_offspring(self, children1: List[Agent], children2: List[Agent]) -> Union[List[Agent], Tuple[List[Agent], List[Agent]]]:
        xp  = self.env.xp
        pop = len(children1)

        if self.return_mode == 'first':
            return children1
        if self.return_mode == 'second':
            return children2
        if self.return_mode == 'both':
            return children1, children2

    
        mask = xp.random.random((pop,)) < 0.5 # (pop,) bool
        mask_cpu = mask.tolist() 
        return [c1 if m else c2
                for c1, c2, m in zip(children1, children2, mask_cpu)]

        
    @property
    def rate(self) -> float:
        return self._rate
    
    @rate.setter
    def rate(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('crossover rate should be a float')
        
        if value < 0.0 or value > 1.0:
            raise e.ValueError('crossover rate should be in interval [0.0, 1.0]')
        
        self._rate = value
    @property
    def return_mode(self) -> str:
        """Which offspring will be returned by the crossover operation.
        """
        return self._return_mode

    @return_mode.setter
    def return_mode(self, value: Literal['first', 'second', 'both', 'random']) -> None:
        # Define allowed options for validation and error reporting
        allowed = ['first', 'second', 'both', 'random']
    
        if value not in allowed:
            # Raising ValueError with a clear explanation of what was expected vs received
            raise ValueError(f"`return_mode` should be one of {allowed}, but got '{value}'.")
    
        self._return_mode = value
          
    @abstractmethod
    def __call__(self, parent1, parent2, *args, **kwargs):
        pass


class ContinuousCrossover(BaseCrossover):
    def __init__(self, rate, gene_rate: float = 0.5, return_mode = 'both'):
        super().__init__(rate, return_mode)
        self.gene_rate = gene_rate

        
    @property
    def gene_rate(self) -> float:
        return self._gene_rate
    
    @gene_rate.setter
    def gene_rate(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('gene rate should be a float')
        
        if value < 0.0 or value > 1.0:
            raise e.ValueError('gene rate should be in interval [0.0, 1.0]')
        
        self._gene_rate = value

class BaseMutation(ABC):
    """Abstract base class for mutation operators."""

    def __init__(self, rate: float = 0.025, ):
        self.rate = rate

        self.env = Environment().set_backend('cpu').set_dtype('float64')


    @property
    def rate(self) -> float:
        return self._rate
    
    @rate.setter
    def rate(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('mutation rate should be a float')
        
        if value < 0.0 or value > 1.0:
            raise e.ValueError('mutation rate should be in interval [0.0, 1.0]')
        
        self._rate = value
        

    @abstractmethod
    def __call__(self, vector, *args, **kwargs):
        pass

class ArithmeticCrossover(ContinuousCrossover):
    """Arithmetic crossover for real-valued vectors."""

    def __init__(self, rate: float = 1.0, gene_rate: float = 1.0, return_mode = 'both'):
        super().__init__(rate, gene_rate, return_mode)
      
    def _arithmetic_positions(self, P1, P2):
        xp = self.env.xp
        active = xp.random.random(P1.shape) < self.gene_rate
        alpha = xp.random.random(P1.shape)
        C1 = xp.where(active, alpha * P1 + (1.0 - alpha) * P2, P1)
        C2 = xp.where(active, alpha * P2 + (1.0 - alpha) * P1, P2)

        return C1, C2
    
    def __call__(self, parent1: Union[Agent, List[Agent]], parent2: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        xp = self.env.xp
        is_batch = isinstance(parent1, list)
        p1_list = parent1 if is_batch else [parent1]
        p2_list = parent2 if is_batch else [parent2]
        pop = len(p1_list)
 
        children1 = [copy.copy(p) for p in p1_list]
        children2 = [copy.copy(p) for p in p2_list]
 
        P1 = xp.stack([p.position.ravel() for p in p1_list])
        P2 = xp.stack([p.position.ravel() for p in p2_list])
        LB = xp.stack([p.lb.ravel() for p in p1_list])
        UB = xp.stack([p.ub.ravel() for p in p1_list])
 
        gate = (xp.random.random((pop,)) < self.rate)[:, None]
 
        C1, C2 = self._arithmetic_positions(P1, P2)
 
        C1 = xp.clip(xp.where(gate, C1, P1), LB, UB)
        C2 = xp.clip(xp.where(gate, C2, P2), LB, UB)
 
        if is_batch:
            C_combined = xp.concatenate([C1, C2], axis=0)
            return BatchList(children1 + children2, tensor=C_combined)
 
        children1[0].position = C1[0].reshape(parent1.position.shape)
        children2[0].position = C2[0].reshape(parent2.position.shape)
        return self._return_offspring(children1, children2)



class GaussianMutation(BaseMutation):
    """Gaussian mutation for real-valued vectors."""

    def __init__(self, rate=0.025, std=0.1):
        super().__init__(rate=rate)
        self.std = std

    def _gaussian_positions(self, X, LB, UB):
        xp = self.env.xp
        active = (
            (xp.random.random(X.shape) < self.rate) &
            (LB != UB)
        )
        noise  = xp.random.normal(0.0, self.std, X.shape)
        X_new  = xp.clip(X + noise, LB, UB)
        return xp.where(active, X_new, X)


    def __call__(self, agent: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        xp = self.env.xp
        is_batch = isinstance(agent, list)
        agents   = agent if is_batch else [agent]
 
        if isinstance(agent, BatchList) and hasattr(agent, 'tensor'):
            X = agent.tensor
        else:
            X = xp.stack([a.position.ravel() for a in agents])
 
        LB = xp.stack([a.lb.ravel() for a in agents])
        UB = xp.stack([a.ub.ravel() for a in agents])
 
        X_new = self._gaussian_positions(X, LB, UB)
 
        if is_batch:
            return BatchList(agents, tensor=X_new)
 
        mutant = copy.copy(agents[0])
        mutant.position = X_new[0].reshape(agents[0].position.shape)
        return mutant



class SBXCrossover(ContinuousCrossover):
    """Simulated Binary Crossover (SBX) for real-valued vectors."""

    def __init__(self, eta = 20, rate: float = 1.0, gene_rate: float = 1.0, return_mode: str = 'random'):
        super().__init__(rate, gene_rate, return_mode)
        self.eta = eta


    def _sbx_positions(self, p1, p2, lb, ub):
        """"""
        xp = self.env.xp

        active = (
            (xp.random.random(p1.shape) < self.gene_rate) &
            (xp.abs(p1 - p2) > 1e-14) &
            (lb != ub)
        )

        y1 = xp.minimum(p1, p2)
        y2 = xp.maximum(p1, p2)
        delta = xp.maximum(y2 - y1, 1e-14)
        rand = xp.random.random(p1.shape)
        exp = 1.0 / (self.eta + 1.0)

        beta1 = 1.0 + 2.0 * (y1 - lb) / delta
        alpha1 = 2.0 - xp.power(beta1, -(self.eta + 1.0))
        betaq1 = xp.where(
            rand <= 1.0 / alpha1,
            xp.power(xp.maximum(rand * alpha1, 1e-14), exp),
            xp.power(xp.maximum(1.0 / (2.0 - rand * alpha1), 1e-14), exp),
        )
        c1 = xp.clip(0.5 * ((y1 + y2) - betaq1 * delta), lb, ub)

        beta2 = 1.0 + 2.0 * (ub - y2) / delta
        alpha2 = 2.0 - xp.power(beta2, -(self.eta + 1.0))
        betaq2 = xp.where(
            rand <= 1.0 / alpha2,
            xp.power(xp.maximum(rand * alpha2, 1e-14), exp),
            xp.power(xp.maximum(1.0 / (2.0 - rand * alpha2), 1e-14), exp),
        )
        c2 = xp.clip(0.5 * ((y1 + y2) + betaq2 * delta), lb, ub)

        swap   = xp.random.random(p1.shape) <= 0.5
        final1 = xp.where(swap, c2, c1)
        final2 = xp.where(swap, c1, c2)

        return xp.where(active, final1, p1), xp.where(active, final2, p2)

    def __call__(self, parent1: Union[Agent, List[Agent]], parent2: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        """
        Vectorized SBX implementation.
        """
     
        # Normalizes for batch
        xp = self.env.xp
        is_batch = isinstance(parent1, list)
        p1_list = parent1 if is_batch else [parent1]
        p2_list = parent2 if is_batch else [parent2]
        pop = len(p1_list)

        children1 = [copy.copy(p) for p in p1_list]
        children2 = [copy.copy(p) for p in p2_list]

        # tensors build

        P1 = xp.stack([p.position.ravel() for p in p1_list])   
        P2 = xp.stack([p.position.ravel() for p in p2_list])
        LB = xp.stack([p.lb.ravel() for p in p1_list])   
        UB = xp.stack([p.ub.ravel() for p in p1_list])   

        # pair-wise
        gate = (xp.random.random((pop,)) < self.rate)[:, None]

        C1, C2 = self._sbx_positions(P1, P2, LB, UB)

        C1 = xp.where(gate, C1, P1)
        C2 = xp.where(gate, C2, P2)

        if is_batch:
            C_combined = xp.concatenate([C1, C2], axis=0)
            return BatchList(children1 + children2, tensor=C_combined)
        else:
            children1[0].position = C1[0].reshape(parent1.position.shape)
            children2[0].position = C2[0].reshape(parent2.position.shape)
            return self._return_offspring(children1, children2)

class OnePointCrossover(BaseCrossover):
    """One-point crossover for binary or real-valued vectors."""
    
    def __init__(self, rate = 1.0, return_mode = 'random'):
        super().__init__(rate, return_mode)
    
    def _one_point_positions(self, P1, P2, LB, UB):
        """
        Vectorized one-point crossover over a batch (pop, n_vars).
        Each pair draws an independent cut point; the split is 1-D along n_vars.
        """
        xp  = self.env.xp
        pop, n_vars = P1.shape
 
        # random cut points in [1, n_vars-1] — one per pair
        # shape (pop, 1) for broadcasting
        points = xp.random.randint(1, n_vars, size=(pop, 1))         
        idx    = xp.arange(n_vars)[None, :]                           
 
        mask = idx < points                                           
 
        C1 = xp.where(mask, P1, P2)
        C2 = xp.where(mask, P2, P1)
 
        C1 = xp.clip(C1, LB, UB)
        C2 = xp.clip(C2, LB, UB)
        return C1, C2

    def __call__(self, parent1: Union[Agent, List[Agent]], parent2: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        xp = self.env.xp
        is_batch = isinstance(parent1, list)
        p1_list = parent1 if is_batch else [parent1]
        p2_list = parent2 if is_batch else [parent2]
        pop = len(p1_list)
 
        children1 = [copy.copy(p) for p in p1_list]
        children2 = [copy.copy(p) for p in p2_list]
 
        P1 = xp.stack([p.position.ravel() for p in p1_list])
        P2 = xp.stack([p.position.ravel() for p in p2_list])
        LB = xp.stack([p.lb.ravel() for p in p1_list])
        UB = xp.stack([p.ub.ravel() for p in p1_list])
 
        n_vars = P1.shape[1]
 
        gate = (xp.random.random((pop,)) < self.rate)[:, None]
 
        if n_vars > 1:
            C1, C2 = self._one_point_positions(P1, P2, LB, UB)
        else:
            C1, C2 = P1, P2
 
        C1 = xp.where(gate, C1, P1)
        C2 = xp.where(gate, C2, P2)
 
        if is_batch:
            C_combined = xp.concatenate([C1, C2], axis=0)
            return BatchList(children1 + children2, tensor=C_combined)
 
        children1[0].position = C1[0].reshape(parent1.position.shape)
        children2[0].position = C2[0].reshape(parent2.position.shape)
        return self._return_offspring(children1, children2)



class BitFlipMutation(BaseMutation):
    """Bit flip mutation for binary vectors."""
    def __init__(self, rate = 0.025):
        super().__init__(rate)

    def _bitflip_positions(self, X):
        xp = self.env.xp
        active = xp.random.random(X.shape) < self.rate
        return xp.where(active, 1 - X, X)


    def __call__(self, agent: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        xp = self.env.xp
        is_batch = isinstance(agent, list)
        agents   = agent if is_batch else [agent]
 
        if isinstance(agent, BatchList) and hasattr(agent, 'tensor'):
            X = agent.tensor
        else:
            X = xp.stack([a.position.ravel() for a in agents])
 
        X_new = self._bitflip_positions(X)
 
        if is_batch:
            return BatchList(agents, tensor=X_new)
 
        mutant = copy.copy(agents[0])
        mutant.position = X_new[0].reshape(agents[0].position.shape)
        return mutant



class PolynomialMutation(BaseMutation):
    """Polynomial mutation for real-valued vectors."""

    def __init__(self, eta : int = 20, rate = 1/30):
        super().__init__(rate)
        self.eta = eta
        
    @property
    def eta(self) -> int:
        return self._eta
    
    @eta.setter
    def eta(self, eta: int) -> None:
        if not isinstance(eta, int):
            raise e.TypeError('`eta` should be an integer')
        if eta <= 0:
            raise e.ValueError('`eta` should be higher than 0')
        
        self._eta = eta


    def _pm_positions(self, x, lb, ub):
        """"""
        xp = self.env.xp
        active = (
            (xp.random.random(x.shape) < self.rate) &
            (lb != ub)
        )

        delta1 = (x - lb)  / xp.maximum(ub - lb, 1e-14)
        delta2 = (ub - x) / xp.maximum(ub - lb, 1e-14)
        rand = xp.random.random(x.shape)
        exp = 1.0 / (self.eta + 1.0)

        val_lo = 2.0 * rand + (1.0 - 2.0 * rand) * xp.power(xp.maximum(1.0 - delta1, 0.0), self.eta + 1.0)
        dq_lo  = xp.power(xp.maximum(val_lo, 1e-14), exp) - 1.0

        val_hi = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xp.power(xp.maximum(1.0 - delta2, 0.0), self.eta + 1.0)
        dq_hi  = 1.0 - xp.power(xp.maximum(val_hi, 1e-14), exp)

        deltaq = xp.where(rand <= 0.5, dq_lo, dq_hi)
        x_new  = xp.clip(x + deltaq * (ub - lb), lb, ub)

        return xp.where(active, x_new, x)
        
    def __call__(self, agent:Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        
        xp = self.env.xp
        is_batch = isinstance(agent, list)

        agents = agent if is_batch else [agent]
        
        # tensor build (pop, n_vars)
        if isinstance(agent, BatchList) and hasattr(agent, 'tensor'):
            X = agent.tensor
        else:
            X = xp.stack([a.position.ravel() for a in agents])

        LB = xp.stack([a.lb.ravel() for a in agents])
        UB = xp.stack([a.ub.ravel() for a in agents])

        X_new = self._pm_positions(X, LB, UB)

        if is_batch:
            return BatchList(agents, tensor=X_new)
        
        agents[0].position = X_new[0].reshape(agent.position.shape)
        return agents[0]
