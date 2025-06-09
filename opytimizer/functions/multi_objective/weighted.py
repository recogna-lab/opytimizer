"""Multi-objective weighted functions.
"""

from typing import List

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class WeightedFunction(Function):
    """A WeightedFunction class used to hold multi-objective weighted functions."""

    def __init__(self, functions: List[callable], weights: List[float]) -> None:
        """Initialization method.

        Args:
            functions: Pointers to functions that will return the fitness value.
            weights: Weights for weighted-sum strategy.

        """

        logger.info("Overriding class: Function -> WeightedFunction.")

        super(WeightedFunction, self).__init__(functions)

        self.functions = functions

        self.weights = weights or []

        logger.debug("Weights: %s", self.weights)
        logger.info("Class overrided.")

    def __call__(self, x: np.ndarray) -> float:
        """Callable to avoid using the `pointer` property.

        Args:
            x: Array of positions.

        Returns:
            (float): Multi-objective weighted function fitness.

        """

        z = 0
        for (f, w) in zip(self.functions, self.weights):
            z += w * f(x)

        return z

    @property
    def weights(self) -> List[float]:
        """Functions' weights."""

        return self._weights

    @weights.setter
    def weights(self, weights: List[float]) -> None:
        if not isinstance(weights, list):
            raise e.TypeError("`weights` should be a list")
        if len(weights) != len(self.functions):
            raise e.SizeError("`weights` should have the same size of `functions`")

        self._weights = weights
