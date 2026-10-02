"""Single-objective functions.
"""

import inspect
from typing import Any, Callable, List, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Function:
    """A Function class used to hold single-objective or multi-objective functions."""

    def __init__(self, pointer: callable, budget: int = None) -> None:
        """Initialization method.

        Args:
            pointer: Pointer to a function or list of functions that will return the fitness value(s).
        """

        logger.info("Creating class: Function.")

        self.pointer = pointer

        if hasattr(pointer, "__name__"):
            self.name = pointer.__name__
        else:
            self.name = pointer.__class__.__name__

        self.n_calls = 0
        self.budget = budget  # None = ilimited

        self.built = True

        logger.debug("Function: %s | Built: %s.", self.name, self.built)
        logger.info("Class created.")

    def __call__(self, x: Any, xp: Any = None) -> np.ndarray:
        """Callable to avoid using the `pointer` property.

        Args:
            x: Array of positions.
            xp: numpy or cupy object

        Returns:
            (np.ndarray): Function fitness value(s).

        """
        if hasattr(x, "ndim"):
            self.n_calls += (x.ndim > 1 and x.shape[1] > 1 and x.shape[0]) or 1
        else:
            # If it is a tree, count as 1 evaluation
            self.n_calls += 1

        if xp is None:
            xp = np

        result = self.pointer(x)
        result = xp.asarray(result)
        return result

    def _accepts_one_arg(self, fn: Callable) -> bool:
        try:
            sig = inspect.signature(fn)
        except (e.ValueError, e.TypeError):
            return False

        try:
            sig.bind(None)  # accepts 1 positional argument
        except e.TypeError:
            return False

        return True

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, budget):
        if budget is not None:
            if not isinstance(budget, int):
                raise e.TypeError("`budget` should be an integer")
            if budget <= 0:
                raise e.ValueError("`budget` should be > 0")

        self._budget = budget

    @property
    def pointer(self) -> callable:
        """callable: Points to the actual function."""

        return self._pointer

    @pointer.setter
    def pointer(self, pointer: Union[Callable, List[Callable]]) -> None:
        items = pointer if isinstance(pointer, list) else [pointer]

        if not items or not all(callable(p) for p in items):
            raise e.TypeError("`pointer` should be a callable or a list of callables")

        if not all(self._accepts_one_arg(p) for p in items):
            raise e.TypeError("`pointer` callables should receive exactly one argument")

        if isinstance(pointer, list):
            funcs = list(pointer)

            def multi_objective(x):
                return np.array([f(x) for f in funcs])

            multi_objective.functions = funcs
            multi_objective.__name__ = ", ".join(f.__name__ for f in funcs)

            self._pointer = multi_objective
        else:
            self._pointer = pointer

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
