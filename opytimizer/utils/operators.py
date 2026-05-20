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
      
    def __call__(self, parent1: Agent, parent2: Agent) -> tuple:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        if self.rate < np.random.random():
            
            x1 = parent1.position
            x2 = parent2.position
            mask = np.random.rand(*x1.shape) < self.gene_rate
            alpha = np.random.uniform(0.0, 1.0, size=x1.shape)

            alpha = r.generate_uniform_random_number(0.0, 1.0, size=x1.shape)

            child1.position[mask] = alpha[mask] * x1[mask] + (1 - alpha[mask]) * x2[mask]
            child2.position[mask] = alpha[mask] * x2[mask] + (1 - alpha[mask]) * x1[mask]

        return self._return_offspring([child1, child2])


class GaussianMutation(BaseMutation):
    """Gaussian mutation for real-valued vectors."""

    def __init__(self, rate=0.025, std=0.1):
        super().__init__(rate=rate)
        self.std = std

    def __call__(self, agent: Agent) -> Agent:
        mutant = copy.deepcopy(agent)
      
        x = agent.position
        lb = agent.lb
        ub = agent.ub

        mask = np.random.rand(*x.shape) < self.rate
        if np.any(mask):
            noise = r.generate_gaussian_random_number(
                mean=0.0, variance=self.std, size=x.shape
            )

            new_position = np.copy(x)
            new_position[mask] += noise[mask]

            mutant.position = np.clip(new_position, lb.reshape(-1,1), ub.reshape(-1,1))
            
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
    def __call__(self, parent1: Agent, parent2: Agent) -> tuple:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        if np.random.rand() < self.rate:
            
            p1 = parent1.position
            p2 = parent2.position
            lb = parent1.lb
            ub = parent1.ub
            
            # chromossome length
            n_vars = p1.shape[0]
            
            if n_vars > 1:
                point = r.generate_integer_random_number(1, p1.shape[0])

        c1 = np.concatenate((p1[:point], p2[point:]))
        c2 = np.concatenate((p2[:point], p1[point:]))
        c1 = np.clip(c1, lb, ub)
        c2 = np.clip(c2, lb, ub)
        
        child1.position = c1
        child2.position = c2
        
        return self._return_offspring([child1, child2])


class BitFlipMutation(BaseMutation):
    """Bit flip mutation for binary vectors."""
    def __init__(self, rate = 0.025):
        super().__init__(rate)

    def __call__(self, agent: Agent) -> Agent:
        mutant = copy.deepcopy(agent)
        x = mutant.position
        mask = np.random.rand(*x.shape) < self.rate
        if np.any(mask):
            x[mask] = 1 - x[mask]
        mutant.position = x
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
