"""Multi-objective weighted functions.
"""

from typing import List, Callable

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.utils import logging
from opytimizer.core.function import Function

logger = logging.get_logger(__name__)


class WeightedFunction(Function):
    """Weighted sum of multiple functions."""
    def __init__(self, functions: List[Callable], weights: List[float]):
        self.functions = [Function(f) for f in functions]
        self.weights = weights
        super().__init__(self._weighted_sum)

    def _weighted_sum(self, x: np.ndarray) -> float:
        return sum(w * f(x) for f, w in zip(self.functions, self.weights))
