import copy
from abc import ABC, abstractmethod

import numpy as np

import opytimizer.math.random as r
from opytimizer.core.agent import Agent
from typing_extensions import Literal
from typing import Union, List
import opytimizer.utils.exception as e

class BaseCrossover(ABC):
    """Abstract base class for crossover operators."""
    
    def __init__(self, rate: float = 1.0, return_mode: Literal['first','second', 'both', 'random'] = 'both') -> None:
        self.rate = rate
        self.return_mode = return_mode
      
    def _return_offspring(self, children: List[Agent]) -> Union[Agent, List[Agent]]:

        if self.return_mode == 'first': return [children[0]]
        elif self.return_mode == 'second': return [children[1]]
        elif self.return_mode == 'both': return children
        else: return [children[np.random.choice([0,1])]]
    
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

    def __init__(self, rate: float = 0.025):
        self.rate = rate

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

    def __init__(self, rate: float = 1.0, gene_rate: float = 0.5, return_mode = 'both'):
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

            mutant.position = np.clip(new_position, lb, ub)
            
        return mutant


class SBXCrossover(ContinuousCrossover):
    """Simulated Binary Crossover (SBX) for real-valued vectors."""

    def __init__(self, eta = 20, rate: float = 1.0, gene_rate: float = 0.5, return_mode: str = 'random'):
        super().__init__(rate, gene_rate, return_mode)
        self.eta = eta

    def __call__(self, parent1: Agent, parent2: Agent) -> tuple:
        """
        Vectorized SBX implementation.
        Expects two parents
        """
     
        # deepcopy to avoid modifying parents by reference
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # ensure column vector shape (N, 1)
        if child1.position.ndim == 1: child1.position = child1.position.reshape(-1, 1)
        if child2.position.ndim == 1: child2.position = child2.position.reshape(-1, 1)

        # global crossover probability check
        if np.random.random() >= self.rate:
            return self._return_offspring([child1, child2])

        
        # flatten arrays for fast numpy broadcasting (N_vars,)
        p1 = parent1.position.flatten()
        p2 = parent2.position.flatten()
        lb = parent1.lb.flatten()
        ub = parent1.ub.flatten()
        
        # determine which genes will be modified
        do_crossover = np.random.random(p1.shape) < self.gene_rate
        
        dist = np.abs(p1-p2)
        too_close = (dist<=1e-14)
        fixed_var = (lb == ub)
        
        do_crossover[too_close] = False
        do_crossover[fixed_var] = False
        
        if not np.any(do_crossover):
            return self._return_offspring([child1, child2])
        
        
        # extract only the values involved in crossover to save computation
        
        y1_val = p1[do_crossover]
        y2_val = p2[do_crossover]
        lb_val = lb[do_crossover]
        ub_val = ub[do_crossover]
        
        
        # order parents
        
        y1 = np.minimum(y1_val, y2_val)
        y2 = np.maximum(y1_val, y2_val)
        
        delta = y2 - y1
        
        # avoid division by zero
        delta = np.maximum(delta, 1e-14)
        
        #generate beta
        rand = np.random.random(y1.shape)
        exponent = 1.0/(self.eta+1.0)
        
        # beta 1 calculation
        
        beta1 = 1.0 + (2.0 * (y1 - lb_val) / delta)
        alpha1 = 2.0 - np.power(beta1, -(self.eta + 1.0))
        
        mask_b1 = rand <= (1.0 / alpha1)
        betaq1 = np.empty_like(y1)
        
        betaq1[mask_b1] = np.power((rand[mask_b1] * alpha1[mask_b1]), exponent)
        betaq1[~mask_b1] = np.power((1.0 / (2.0 - rand[~mask_b1] * alpha1[~mask_b1])), exponent)
        
        # child1 value
        c1_new = 0.5 * ((y1+y2) - betaq1 * delta)
        
        
        
        # beta 2 calculation
        
        beta2 = 1.0 + (2.0 * (ub_val - y2) / delta)
        alpha2 = 2.0 - np.power(beta2, -(self.eta + 1.0))
        
        mask_b2 = rand <= (1.0 / alpha2)
        betaq2 = np.empty_like(y1)
        
        betaq2[mask_b2] = np.power((rand[mask_b2] * alpha2[mask_b2]), exponent)
        betaq2[~mask_b2] = np.power((1.0 / (2.0 - rand[~mask_b2] * alpha2[~mask_b2])), exponent)
        
        # child2
        c2_new = 0.5 * ((y1 + y2) + betaq2 * delta)
        
        # bound reapir
        
        c1_new = np.clip(c1_new, lb_val, ub_val)
        c2_new = np.clip(c2_new, lb_val, ub_val)
        
        # random swap (to avoid bias)
        
        swap_mask = np.random.random(c1_new.shape) <= 0.5
        final_c1_vals = np.where(swap_mask, c2_new, c1_new)
        final_c2_vals = np.where(swap_mask, c1_new, c2_new)
        
        temp_c1 = child1.position.flatten()
        temp_c2 = child2.position.flatten()
        
        temp_c1[do_crossover] = final_c1_vals
        temp_c2[do_crossover] = final_c2_vals
        
        
        child1.position = temp_c1.reshape(-1, 1)
        child2.position = temp_c2.reshape(-1, 1)
        
        
        return self._return_offspring([child1, child2])


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
        
    def __call__(self, agent: Agent) -> Agent:
        mutant = copy.deepcopy(agent)
        original_shape = mutant.position.shape
        # flatten for vectorization
        x = mutant.position.flatten()
        lb = mutant.lb.flatten()
        ub = mutant.ub.flatten()
            
        # probability mask: determine which genes mutate
        mask = np.random.random(x.shape) < self.rate
            
        #  do not mutate fixed variables
        mask[lb == ub] = False
            
        # optimization: Only compute if at least one gene mutates
        if np.any(mask):
            x_mut = x[mask]
            lb_mut = lb[mask]
            ub_mut = ub[mask]
                
            delta1 = (x_mut - lb_mut) / (ub_mut - lb_mut + 1e-14)
            delta2 = (ub_mut - x_mut) / (ub_mut - lb_mut + 1e-14)
                
            rand = np.random.random(x_mut.shape)
            mut_pow = 1.0 / (self.eta + 1.0)
            deltaq = np.zeros_like(x_mut)
                
                
            lower = rand <= 0.5
            if np.any(lower):
                xy = 1.0 - delta1[lower]
                val = 2.0 * rand[lower] + (1.0 - 2.0 * rand[lower]) * np.power(xy, (self.eta + 1.0))
                deltaq[lower] = np.power(val, mut_pow) - 1.0
                    
            upper = ~lower
            if np.any(upper):
                xy = 1.0 - delta2[upper]
                val = 2.0 * (1.0 - rand[upper]) + 2.0 * (rand[upper] - 0.5) * np.power(xy, (self.eta + 1.0))
                deltaq[upper] = 1.0 - np.power(val, mut_pow)
                    
            # apply displacement
            y = x_mut + deltaq * (ub_mut - lb_mut)
            x[mask] = np.clip(y, lb_mut, ub_mut)
                
            # reshape back to column vector
            mutant.position = x.reshape(original_shape)
            
        return mutant