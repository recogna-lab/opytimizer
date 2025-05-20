"""Single-objective functions.
"""

from inspect import signature

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Function:
    """A Function class used to hold single-objective or multi-objective functions."""

    def __init__(self, pointer: callable) -> None:
        """Initialization method.

        Args:
            pointer: Pointer to a function or list of functions that will return the fitness value(s).
        """

        logger.info("Creating class: Function.")

        if isinstance(pointer, list):
            for f in pointer:
                if not callable(f):
                    raise e.TypeError("All elements in the list must be callable")
            self.pointer = lambda x: [f(x) for f in pointer]
            self.n_objectives = len(pointer)
            self.name = "MultiObjectiveFunction"
        else:
            self.pointer = pointer
            self.n_objectives = 1
            if hasattr(pointer, "__name__"):
                self.name = pointer.__name__
            else:
                self.name = pointer.__class__.__name__

        self.built = True

        logger.debug("Function: %s | Built: %s.", self.name, self.built)
        logger.info("Class created.")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Callable to avoid using the `pointer` property.

        Args:
            x: Array of positions.

        Returns:
            (np.ndarray): Function fitness value(s).

        """

        result = self.pointer(x)
        if isinstance(result, (int, float)):
            return np.array([result])
        return np.array(result)

    @property
    def pointer(self) -> callable:
        """callable: Points to the actual function."""

        return self._pointer

    @pointer.setter
    def pointer(self, pointer: callable) -> None:
        if not callable(pointer):
            raise e.TypeError("`pointer` should be a callable")
        if len(signature(pointer).parameters) > 1:
            raise e.ArgumentError("`pointer` should only have 1 argument")

        self._pointer = pointer

    @property
    def n_objectives(self) -> int:
        """int: Number of objectives."""

        return self._n_objectives

    @n_objectives.setter
    def n_objectives(self, n_objectives: int) -> None:
        if not isinstance(n_objectives, int):
            raise e.TypeError("`n_objectives` should be an integer")
        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be > 0")

        self._n_objectives = n_objectives

    @property
    def name(self) -> str:
        """Name of the function."""

        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise e.TypeError("`name` should be a string")

        self._name = name

    @property
    def built(self) -> bool:
        """Indicates whether the function is built."""

        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        self._built = built
