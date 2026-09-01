import copy
from abc import ABC, abstractmethod
import numpy as np

from typing import Union, List, Tuple, Any
from opytimizer.core.agent import Agent
from opytimizer.core import Environment
import opytimizer.utils.exception as e


# BASE ABSTRACT CLASSES 


class BaseCrossover(ABC):
    """Abstract base class for CPU crossover operators handling Agent objects."""
    
    def __init__(self, rate: float = 1.0, n_offspring: int = 2) -> None:
        self.rate = rate
        self.n_offspring = n_offspring

    def _return_offspring(self, children1: Union[Agent, List[Agent]], children2: Union[Agent, List[Agent]]) -> Union[List[Agent], Tuple[List[Agent], List[Agent]]]:
        if self.n_offspring == 1:
            return children1
        else:
            return children1, children2

        
    @property
    def rate(self) -> float:
        return self._rate
    
    @rate.setter
    def rate(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('Crossover rate should be a float')
        if value < 0.0 or value > 1.0:
            raise e.ValueError('Crossover rate should be in interval [0.0, 1.0]')
        self._rate = value

    @property
    def n_offspring(self) -> int:
        return self._n_offspring

    @n_offspring.setter
    def n_offspring(self, n: int) -> None:
        if not isinstance(n, int):
            raise e.TypeError('`n_offspring` should be an integer.')
        if n not in (1, 2):
            raise e.ValueError('Error: `n_offspring` possible values are: {1, 2}')
        
        self._n_offspring = n
          
    @abstractmethod
    def __call__(self, parent1, parent2, *args, **kwargs):
        pass


class ContinuousCrossover(BaseCrossover):
    """Abstract base class for continuous space CPU crossovers."""
    def __init__(self, rate: float, gene_rate: float = 0.5, n_offspring: int = 2):
        super().__init__(rate, n_offspring)
        self.gene_rate = gene_rate

    @property
    def gene_rate(self) -> float:
        return self._gene_rate
    
    @gene_rate.setter
    def gene_rate(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('Gene rate should be a float')
        if value < 0.0 or value > 1.0:
            raise e.ValueError('Gene rate should be in interval [0.0, 1.0]')
        self._gene_rate = value


class BaseMutation(ABC):
    """Abstract base class for CPU mutation operators handling Agent objects."""

    def __init__(self, rate: float = 0.025):
        self.rate = rate

    @property
    def rate(self) -> float:
        return self._rate
    
    @rate.setter
    def rate(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('Mutation rate should be a float')
        if value < 0.0 or value > 1.0:
            raise e.ValueError('Mutation rate should be in interval [0.0, 1.0]')
        self._rate = value
        
    @abstractmethod
    def __call__(self, agent, *args, **kwargs):
        pass



# CPU OPERATORS 

class ArithmeticCrossover(ContinuousCrossover):
    """Arithmetic crossover for real-valued vectors operating on Agents (CPU)."""

    def __init__(self, rate: float = 1.0, gene_rate: float = 1.0, n_offspring: int = 2):
        super().__init__(rate, gene_rate, n_offspring)
      
    def _arithmetic_positions(self, P1, P2):
        active = np.random.random(P1.shape) < self.gene_rate
        alpha = np.random.random(P1.shape)
        C1 = np.where(active, alpha * P1 + (1.0 - alpha) * P2, P1)
        C2 = np.where(active, alpha * P2 + (1.0 - alpha) * P1, P2)
        return C1, C2
    
    def __call__(self, parent1: Union[Agent, List[Agent]], parent2: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        is_batch = isinstance(parent1, list)

        p1_list = parent1 if is_batch else [parent1]
        p2_list = parent2 if is_batch else [parent2]
        pop = len(p1_list)
 
        children1 = [copy.copy(p) for p in p1_list]
        children2 = [copy.copy(p) for p in p2_list]
 
        P1 = np.stack([p.position.ravel() for p in p1_list])
        P2 = np.stack([p.position.ravel() for p in p2_list])
        LB = np.stack([p.lb.ravel() for p in p1_list])
        UB = np.stack([p.ub.ravel() for p in p1_list])
 
        gate = (np.random.random((pop,)) < self.rate)[:, None]
        C1, C2 = self._arithmetic_positions(P1, P2)
 
        C1 = np.clip(np.where(gate, C1, P1), LB, UB)
        C2 = np.clip(np.where(gate, C2, P2), LB, UB)
 
        if is_batch:
            return children1 + children2
 
        children1[0].position = C1[0].reshape(parent1.position.shape)
        children2[0].position = C2[0].reshape(parent2.position.shape)
        return self._return_offspring(children1, children2)


class GaussianMutation(BaseMutation):
    """Gaussian mutation for real-valued vectors operating on Agents (CPU)."""

    def __init__(self, rate: float = 0.025, std: float = 0.1):
        super().__init__(rate=rate)
        self.std = std

    def _gaussian_positions(self, X, LB, UB):
        active = (np.random.random(X.shape) < self.rate) & (LB != UB)
        noise = np.random.normal(0.0, self.std, X.shape)
        X_new = np.clip(X + noise, LB, UB)
        return np.where(active, X_new, X)

    def __call__(self, agent: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        is_batch = isinstance(agent, list)


        agents = agent if is_batch else [agent]
 
        X = np.stack([a.position.ravel() for a in agents])
        LB = np.stack([a.lb.ravel() for a in agents])
        UB = np.stack([a.ub.ravel() for a in agents])
 
        X_new = self._gaussian_positions(X, LB, UB)
 
        if is_batch:
            for idx, a in enumerate(agents):
                a.position = X_new[idx].reshape(a.position.shape)
            return agents
 
        mutant = copy.copy(agents[0])
        mutant.position = X_new[0].reshape(agents[0].position.shape)
        return mutant


class SBXCrossover(ContinuousCrossover):
    """Simulated Binary Crossover (SBX) operating on Agents (CPU)."""

    def __init__(self, eta: int = 20, rate: float = 1.0, gene_rate: float = 1.0, n_offspring: int = 2):
        super().__init__(rate, gene_rate, n_offspring)
        self.eta = eta

    def _sbx_positions(self, p1, p2, lb, ub):
        active = (np.random.random(p1.shape) < self.gene_rate) & (np.abs(p1 - p2) > 1e-14) & (lb != ub)
        y1 = np.minimum(p1, p2)
        y2 = np.maximum(p1, p2)
        delta = np.maximum(y2 - y1, 1e-14)
        rand = np.random.random(p1.shape)
        exp = 1.0 / (self.eta + 1.0)

        beta1 = 1.0 + 2.0 * (y1 - lb) / delta
        alpha1 = 2.0 - np.power(beta1, -(self.eta + 1.0))
        betaq1 = np.where(rand <= 1.0 / alpha1, np.power(np.maximum(rand * alpha1, 1e-14), exp), np.power(np.maximum(1.0 / (2.0 - rand * alpha1), 1e-14), exp))
        c1 = np.clip(0.5 * ((y1 + y2) - betaq1 * delta), lb, ub)

        beta2 = 1.0 + 2.0 * (ub - y2) / delta
        alpha2 = 2.0 - np.power(beta2, -(self.eta + 1.0))
        betaq2 = np.where(rand <= 1.0 / alpha2, np.power(np.maximum(rand * alpha2, 1e-14), exp), np.power(np.maximum(1.0 / (2.0 - rand * alpha2), 1e-14), exp))
        c2 = np.clip(0.5 * ((y1 + y2) + betaq2 * delta), lb, ub)

        swap = np.random.random(p1.shape) <= 0.5
        final1 = np.where(swap, c2, c1)
        final2 = np.where(swap, c1, c2)

        return np.where(active, final1, p1), np.where(active, final2, p2)

    def __call__(self, parent1: Union[Agent, List[Agent]], parent2: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        is_batch = isinstance(parent1, list) 

        p1_list = parent1 if is_batch else [parent1]
        p2_list = parent2 if is_batch else [parent2]
        pop = len(p1_list)

        children1 = [copy.copy(p) for p in p1_list]
        children2 = [copy.copy(p) for p in p2_list]

        P1 = np.stack([p.position.ravel() for p in p1_list])   
        P2 = np.stack([p.position.ravel() for p in p2_list])
        LB = np.stack([p.lb.ravel() for p in p1_list])   
        UB = np.stack([p.ub.ravel() for p in p1_list])   

        gate = (np.random.random((pop,)) < self.rate)[:, None]
        C1, C2 = self._sbx_positions(P1, P2, LB, UB)

        C1 = np.where(gate, C1, P1)
        C2 = np.where(gate, C2, P2)

        if is_batch:
            for idx in range(pop):
                children1[idx].position = C1[idx].reshape(p1_list[idx].position.shape)
                children2[idx].position = C2[idx].reshape(p2_list[idx].position.shape)
            return children1 + children2
        else:
            children1[0].position = C1[0].reshape(parent1.position.shape)
            children2[0].position = C2[0].reshape(parent2.position.shape)
            return self._return_offspring(children1, children2)


class OnePointCrossover(BaseCrossover):
    """One-point crossover operating on Agents (CPU)."""
    
    def __init__(self, rate: float = 1.0, n_offspring: int = 2):
        super().__init__(rate, n_offspring)
    
    def _one_point_positions(self, P1, P2, LB, UB):
        pop, n_vars = P1.shape
        points = np.random.randint(1, n_vars, size=(pop, 1))         
        idx = np.arange(n_vars)[None, :]                           
        mask = idx < points                                           
 
        C1 = np.clip(np.where(mask, P1, P2), LB, UB)
        C2 = np.clip(np.where(mask, P2, P1), LB, UB)
        return C1, C2

    def __call__(self, parent1: Union[Agent, List[Agent]], parent2: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        is_batch = isinstance(parent1, list)


        p1_list = parent1 if is_batch else [parent1]
        p2_list = parent2 if is_batch else [parent2]
        pop = len(p1_list)
 
        children1 = [copy.copy(p) for p in p1_list]
        children2 = [copy.copy(p) for p in p2_list]
 
        P1 = np.stack([p.position.ravel() for p in p1_list])
        P2 = np.stack([p.position.ravel() for p in p2_list])
        LB = np.stack([p.lb.ravel() for p in p1_list])
        UB = np.stack([p.ub.ravel() for p in p1_list])
 
        n_vars = P1.shape[1]
        gate = (np.random.random((pop,)) < self.rate)[:, None]
 
        if n_vars > 1:
            C1, C2 = self._one_point_positions(P1, P2, LB, UB)
        else:
            C1, C2 = P1, P2
 
        C1 = np.where(gate, C1, P1)
        C2 = np.where(gate, C2, P2)
 
        if is_batch:
            for idx in range(pop):
                children1[idx].position = C1[idx].reshape(p1_list[idx].position.shape)
                children2[idx].position = C2[idx].reshape(p2_list[idx].position.shape)
            return children1 + children2
 
        children1[0].position = C1[0].reshape(parent1.position.shape)
        children2[0].position = C2[0].reshape(parent2.position.shape)
        return self._return_offspring(children1, children2)


class BitFlipMutation(BaseMutation):
    """Bit flip mutation for binary vectors operating on Agents (CPU)."""
    def __init__(self, rate: float = 0.025):
        super().__init__(rate)

    def _bitflip_positions(self, X):
        active = np.random.random(X.shape) < self.rate
        return np.where(active, 1 - X, X)

    def __call__(self, agent: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        is_batch = isinstance(agent, list)

        agents = agent if is_batch else [agent]
 
        X = np.stack([a.position.ravel() for a in agents])
        X_new = self._bitflip_positions(X)
 
        if is_batch:
            for idx, a in enumerate(agents):
                a.position = X_new[idx].reshape(a.position.shape)
            return agents
 
        mutant = copy.copy(agents[0])
        mutant.position = X_new[0].reshape(agents[0].position.shape)
        return mutant


class PolynomialMutation(BaseMutation):
    """Polynomial mutation for real-valued vectors operating on Agents (CPU)."""

    def __init__(self, eta: int = 20, rate: float = 1/30):
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
        active = (np.random.random(x.shape) < self.rate) & (lb != ub)
        delta1 = (x - lb) / np.maximum(ub - lb, 1e-14)
        delta2 = (ub - x) / np.maximum(ub - lb, 1e-14)
        rand = np.random.random(x.shape)
        exp = 1.0 / (self.eta + 1.0)

        val_lo = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(np.maximum(1.0 - delta1, 0.0), self.eta + 1.0)
        dq_lo = np.power(np.maximum(val_lo, 1e-14), exp) - 1.0

        val_hi = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(np.maximum(1.0 - delta2, 0.0), self.eta + 1.0)
        dq_hi = 1.0 - np.power(np.maximum(val_hi, 1e-14), exp)

        deltaq = np.where(rand <= 0.5, dq_lo, dq_hi)
        x_new = np.clip(x + deltaq * (ub - lb), lb, ub)
        return np.where(active, x_new, x)
        
    def __call__(self, agent: Union[Agent, List[Agent]]) -> Union[Agent, List[Agent]]:
        is_batch = isinstance(agent, list)
        
        agents = agent if is_batch else [agent]
        
        X = np.stack([a.position.ravel() for a in agents])
        LB = np.stack([a.lb.ravel() for a in agents])
        UB = np.stack([a.ub.ravel() for a in agents])

        X_new = self._pm_positions(X, LB, UB)

        if is_batch:
            for idx, a in enumerate(agents):
                a.position = X_new[idx].reshape(a.position.shape)
            return agents
        
        agents[0].position = X_new[0].reshape(agent.position.shape)
        return agents[0]



# TENSORIZED GPU OPERATORS

class ArithmeticCrossoverTensor:
    """
    Highly parallelized Arithmetic Crossover executing on GPU.
    """
    def __init__(self, env: Environment, rate: float = 1.0, gene_rate: float = 1.0):
        self.env = env
        self.rate = rate
        self.gene_rate = gene_rate

    def __call__(self, X1: Any, X2: Any, lb: Any, ub: Any) -> Tuple[Any, Any]:
        """
        Args:
            X1, X2: Coordinate tensors of shape (N, D).
            lb, ub: Boundary constraint tensors of shape (D,) or (N, D).
        Returns:
            Tuple of generated offspring tensors (C1, C2) of shape (N, D).
        """
        xp = self.env.xp
        pop = X1.shape[0]
        
        # Binary gate vector determining which parent pairs cross over
        gate = (xp.random.random((pop, 1)) < self.rate)
        # Element-wise array determining which variables get mixed
        active = xp.random.random(X1.shape) < self.gene_rate
        alpha = xp.random.random(X1.shape)
        
        C1_pos = xp.where(active, alpha * X1 + (1.0 - alpha) * X2, X1)
        C2_pos = xp.where(active, alpha * X2 + (1.0 - alpha) * X1, X2)
        
        # Keep old positions if cross-over rule is skipped, then clip to bounds
        C1 = xp.clip(xp.where(gate, C1_pos, X1), lb, ub)
        C2 = xp.clip(xp.where(gate, C2_pos, X2), lb, ub)
        return C1, C2


class GaussianMutationTensor:
    """
    Fully vectorized Gaussian Mutation running directly on the GPU.
    """
    def __init__(self, env: Environment, rate: float = 0.025, std: float = 0.1):
        self.env = env
        self.rate = rate
        self.std = std

    def __call__(self, X: Any, lb: Any, ub: Any) -> Any:
        """
        Args:
            X: Matrix coordinate tensor of shape (N, D).
            lb, ub: Boundary constraint tensors of shape (D,) or (N, D).
        Returns:
            Mutated coordinate tensor of shape (N, D).
        """
        xp = self.env.xp
        # Construct an active logical mask for indices matching mutation parameters
        active = (xp.random.random(X.shape) < self.rate) & (lb != ub)
        noise = xp.random.normal(0.0, self.std, X.shape)
        X_new = xp.clip(X + noise, lb, ub)
        
        return xp.where(active, X_new, X)


class SBXCrossoverTensor:
    """
    Simulated Binary Crossover (SBX) re-architected for tensorized operations.
    """
    def __init__(self, env: Environment, eta: int = 20, rate: float = 1.0, gene_rate: float = 1.0):
        self.env = env
        self.eta = eta
        self.rate = rate
        self.gene_rate = gene_rate

    def __call__(self, X1: Any, X2: Any, lb: Any, ub: Any) -> Tuple[Any, Any]:
        """
        Args:
            X1, X2: Coordinate tensors of shape (N, D).
            lb, ub: Boundary constraint tensors of shape (D,) or (N, D).
        Returns:
            Tuple of generated offspring tensors (C1, C2) of shape (N, D).
        """
        xp = self.env.xp
       
        pop = X1.shape[0]
        gate = (xp.random.random((pop, 1)) < self.rate)
        
        active = (
            (xp.random.random(X1.shape) < self.gene_rate) &
            (xp.abs(X1 - X2) > 1e-14) &
            (lb != ub)
        )
        
        y1 = xp.minimum(X1, X2)
        y2 = xp.maximum(X1, X2)
        delta = xp.maximum(y2 - y1, 1e-14)
        rand = xp.random.random(X1.shape)
        exp = 1.0 / (self.eta + 1.0)
        
        # Parallel computation of distribution boundaries (Beta Q)
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
        
        # Matrix masking to perform element-wise swapping
        swap = xp.random.random(X1.shape) <= 0.5
        final1 = xp.where(swap, c2, c1)
        final2 = xp.where(swap, c1, c2)
        
        C1_pos = xp.where(active, final1, X1)
        C2_pos = xp.where(active, final2, X2)
        
        C1 = xp.where(gate, C1_pos, X1)
        C2 = xp.where(gate, C2_pos, X2)
        return C1, C2


class OnePointCrossoverTensor:
    """
    Vectorized One-Point Crossover for batches of multidimensional coordinate tensors.
    """
    def __init__(self, env: Environment, rate: float = 1.0):
        self.env = env
        self.rate = rate

    def __call__(self, X1: Any, X2: Any, lb: Any, ub: Any) -> Tuple[Any, Any]:
        """
        Args:
            X1, X2: Coordinate tensors of shape (N, D).
            lb, ub: Boundary constraint tensors of shape (D,) or (N, D).
        Returns:
            Tuple of generated offspring tensors (C1, C2) of shape (N, D).
        """
        xp = self.env.xp
    
        pop, n_vars = X1.shape
        gate = (xp.random.random((pop, 1)) < self.rate)
        
        if n_vars > 1:
            # Independent cut points created using broadcasting index grids
            points = xp.random.randint(1, n_vars, size=(pop, 1))
            idx = xp.arange(n_vars)[None, :]
            mask = idx < points
            C1_pos = xp.where(mask, X1, X2)
            C2_pos = xp.where(mask, X2, X1)
        else:
            C1_pos, C2_pos = X1, X2
            
        C1 = xp.clip(xp.where(gate, C1_pos, X1), lb, ub)
        C2 = xp.clip(xp.where(gate, C2_pos, X2), lb, ub)
        return C1, C2


class BitFlipMutationTensor:
    """
    High-speed binary tensor flip operations on GPU memory.
    """
    def __init__(self, env: Environment, rate: float = 0.025):
        self.env = env
        self.rate = rate

    def __call__(self, X: Any, lb: Any = None, ub: Any = None) -> Any:
        """
        Args:
            X: Matrix coordinate tensor of shape (N, D) composed of binary spaces.
            lb, ub: Unused optional bounds for layout consistency.
        Returns:
            Mutated binary tensor of shape (N, D).
        """
        xp = self.env.xp
        active = xp.random.random(X.shape) < self.rate
        return xp.where(active, 1 - X, X)


class PolynomialMutationTensor:
    """
    Polynomial Mutation designed for tensorized execution.
    """
    def __init__(self, env: Environment, eta: int = 20, rate: float = 1/30):
        self.env = env
        self.eta = eta
        self.rate = rate

    def __call__(self, X: Any, lb: Any, ub: Any) -> Any:
        """
        Args:
            X: Coordinate tensor of shape (N, D).
            lb, ub: Boundary constraint tensors of shape (D,) or (N, D).
        Returns:
            Mutated coordinate tensor of shape (N, D).
        """
      
        xp = self.env.xp
        
        active = (xp.random.random(X.shape) < self.rate) &(lb != ub)
        
        delta1 = (X - lb) / xp.maximum(ub - lb, 1e-14)
        delta2 = (ub - X) / xp.maximum(ub - lb, 1e-14)
        rand = xp.random.random(X.shape)
        exp = 1.0 / (self.eta + 1.0)
        
        val_lo = 2.0 * rand + (1.0 - 2.0 * rand) * xp.power(xp.maximum(1.0 - delta1, 0.0), self.eta + 1.0)
        dq_lo = xp.power(xp.maximum(val_lo, 1e-14), exp) - 1.0
        
        val_hi = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xp.power(xp.maximum(1.0 - delta2, 0.0), self.eta + 1.0)
        dq_hi = 1.0 - xp.power(xp.maximum(val_hi, 1e-14), exp)
        
        deltaq = xp.where(rand <= 0.5, dq_lo, dq_hi)
        X_new = xp.clip(X + deltaq * (ub - lb), lb, ub)
        
        return xp.where(active, X_new, X)