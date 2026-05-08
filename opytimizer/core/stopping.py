"""Stopping criteria for the optimization loop."""

from abc import ABC, abstractmethod
from typing import List, Optional
from tqdm import tqdm
class _StoppingCriterion(ABC):
    """Base Class.  `should_stop` must be implemented in each subclass """

    def __init__(self):
        self.pbar = None

    @abstractmethod
    def should_stop(self, opt) -> bool:
        """Return True when the criterio was satisfied"""

    def reset(self) -> None:
        """Internal State reinicialization."""
        if self.pbar:
            self.pbar.reset()

    def init_pbar(self, position: int) -> None:
        """Progress Bar initialization."""
        pass

    def update_pbar(self, opt) -> None:
        """Update the Progress Bar to the current optimization state."""
        pass

    def close_pbar(self) -> None:
        """Closes the Progress Bar safely """
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None



class MaxIterations(_StoppingCriterion):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    
    def should_stop(self, opt) -> bool:
        return opt.total_iterations >= self.n
    

    def init_pbar(self, position: int) -> None:
        self.pbar = tqdm(total=self.n, desc="Max iterations", position=position, leave=True, ascii=True)

    def update_pbar(self, opt) -> None:
        if self.pbar:
            self.pbar.n = opt.total_iterations
            self.pbar.refresh()
        
        
    
class MaxEvaluations(_StoppingCriterion):
    """Soft check: verified at end of each iteration.
        For exact enforcement, use Function(budget=n) instead."""
    
    def __init__(self, n: int) -> None:
       super().__init__()
       self.n =n

    def should_stop(self, opt) -> bool:
        return opt.n_evals >= self.n
    
    def init_pbar(self, position) -> None:
        self.pbar = tqdm(total=self.n, desc="Max function evaluations", position=position, leave=True, ascii=True)

    def update_pbar(self, opt) -> None:
        if self.pbar:
            self.pbar.n = opt.n_evals
            self.pbar.refresh()

class NoImprovement(_StoppingCriterion):
    """`patience` non-improvement iterations"""

    def __init__(self, patience: int, min_delta: float = 1e-8) ->None:
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0


    def should_stop(self, opt) -> bool:
        if opt.space.n_objectives != 1:
            return False
        
        current = opt.space.best_agent.fit
        if self._best - current > self.min_delta:
            self._best = current
            self._counter = 0

        else:
            self._counter += 1

        return self._counter >= self.patience
    
    def reset(self) -> None:
        self._best = float("inf")
        self._counter = 0



    def init_pbar(self, position) -> None:
        self.pbar = tqdm(total=self.patience, desc="No improvement iterations", position=position, leave=True, ascii=True)

    def update_pbar(self, opt) -> None:
        if self.pbar:
            self.pbar.n = self._counter
            self.pbar.refresh()





class _StoppingVessel:
    """Criteria aggregation."""

    def __init__(self, criteria: Optional[List[_StoppingCriterion]]) -> None:
        self._criteria = criteria or []

    def should_stop(self, opt) -> bool:
        return any(c.should_stop(opt) for c in self._criteria)
    
    def reset(self) -> None:
        for c in self._criteria:
            c.reset()

    def init_pbars(self) -> None:
        for i, c in enumerate(self._criteria):
            c.init_pbar(position=i)

    def update_pbars(self, opt) -> None:
        for c in self._criteria:
            c.update_pbar(opt)

    def set_postfix(self, **kwargs) -> None:
        for c in self._criteria:
            if c.pbar:
                c.pbar.set_postfix(**kwargs)

    def close_pbars(self) -> None:
        for c in self._criteria:
            c.close_pbar()