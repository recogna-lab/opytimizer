"""HyperHeuristic.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class HyperHeuristic(Optimizer):
    """
    A HyperHeuristic class that manages multiple low-level optimizers
    and provides high-level strategies for algorithm selection and adaptation.

    It supports both single-objective and multi-objective optimization by allowing
    a custom performance_metric function to be passed (e.g., min_fitness, hypervolume, etc).

    It also supports parameter adaptation and strategy adaptation mechanisms.
    """

    def __init__(
        self,
        optimizers: Optional[List[Optimizer]] = None,
        performance_metric: Optional[Callable[[Any], float]] = None,
    ) -> None:
        """Initialization method.

        Args:
            optimizers: List of low-level optimizers to be managed.
            performance_metric: Function to evaluate optimizer performance.
        """
        logger.info("Overriding class: Optimizer -> HyperHeuristic.")
        super().__init__()
        self.optimizers = optimizers or []
        self.current_optimizer = None
        self.optimizer_history = []
        self.performance_history = {}
        self.selection_count = {}
        self.iteration = 0
        self.performance_metric = performance_metric

        # Initialize performance tracking for each optimizer
        if self.optimizers:
            for optimizer in self.optimizers:
                optimizer_name = optimizer.__class__.__name__
                self.performance_history[optimizer_name] = []
                self.selection_count[optimizer_name] = 0

        # Mark as built since we have all required components
        self.built = True

        logger.info("Class overrided.")

    @property
    def optimizers(self) -> List[Optimizer]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: List[Optimizer]) -> None:
        if not isinstance(optimizers, list):
            raise e.TypeError("`optimizers` should be a list")
        for optimizer in optimizers:
            if not isinstance(optimizer, Optimizer):
                raise e.TypeError("All optimizers should be Optimizer instances")
        self._optimizers = optimizers

    @property
    def current_optimizer(self) -> Optional[Optimizer]:
        return self._current_optimizer

    @current_optimizer.setter
    def current_optimizer(self, current_optimizer: Optional[Optimizer]) -> None:
        if current_optimizer is not None and not isinstance(
            current_optimizer, Optimizer
        ):
            raise e.TypeError("`current_optimizer` should be an Optimizer instance")
        self._current_optimizer = current_optimizer

    @property
    def optimizer_history(self) -> List[str]:
        return self._optimizer_history

    @optimizer_history.setter
    def optimizer_history(self, optimizer_history: List[str]) -> None:
        if not isinstance(optimizer_history, list):
            raise e.TypeError("`optimizer_history` should be a list")
        self._optimizer_history = optimizer_history

    @property
    def performance_history(self) -> Dict[str, List[float]]:
        return self._performance_history

    @performance_history.setter
    def performance_history(self, performance_history: Dict[str, List[float]]) -> None:
        if not isinstance(performance_history, dict):
            raise e.TypeError("`performance_history` should be a dictionary")
        self._performance_history = performance_history

    @property
    def selection_count(self) -> Dict[str, int]:
        return self._selection_count

    @selection_count.setter
    def selection_count(self, selection_count: Dict[str, int]) -> None:
        if not isinstance(selection_count, dict):
            raise e.TypeError("`selection_count` should be a dictionary")
        self._selection_count = selection_count

    @property
    def iteration(self) -> int:
        return self._iteration

    @iteration.setter
    def iteration(self, iteration: int) -> None:
        if not isinstance(iteration, int):
            raise e.TypeError("`iteration` should be an integer")
        if iteration < 0:
            raise e.ValueError("`iteration` should be >= 0")
        self._iteration = iteration

    def add_optimizer(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, Optimizer):
            raise e.TypeError("`optimizer` should be an Optimizer instance")
        self.optimizers.append(optimizer)
        optimizer_name = optimizer.__class__.__name__
        self.performance_history[optimizer_name] = []
        self.selection_count[optimizer_name] = 0
        logger.debug("Added optimizer: %s.", optimizer_name)

    def remove_optimizer(self, optimizer_name: str) -> None:
        if not isinstance(optimizer_name, str):
            raise e.TypeError("`optimizer_name` should be a string")
        for i, optimizer in enumerate(self.optimizers):
            if optimizer.__class__.__name__ == optimizer_name:
                del self.optimizers[i]
                if optimizer_name in self.performance_history:
                    del self.performance_history[optimizer_name]
                if optimizer_name in self.selection_count:
                    del self.selection_count[optimizer_name]
                break
        logger.debug("Removed optimizer: %s.", optimizer_name)

    def select_optimizer(self, space: Space, function: Function) -> Optimizer:
        if not self.optimizers:
            raise e.ValueError("No optimizers available for selection")
        selected_index = self.iteration % len(self.optimizers)
        selected_optimizer = self.optimizers[selected_index]
        optimizer_name = selected_optimizer.__class__.__name__
        self.selection_count[optimizer_name] += 1
        self.optimizer_history.append(optimizer_name)
        logger.debug("Selected optimizer: %s.", optimizer_name)
        return selected_optimizer

    def update_performance(self, optimizer: Optimizer, space: Space) -> None:
        if not isinstance(optimizer, Optimizer):
            raise e.TypeError("`optimizer` should be an Optimizer instance")
        optimizer_name = optimizer.__class__.__name__
        if optimizer_name not in self.performance_history:
            self.performance_history[optimizer_name] = []
        # Use the custom performance metric if provided
        if self.performance_metric is not None:
            performance = self.performance_metric(space)
            self.performance_history[optimizer_name].append(performance)
            logger.debug("Updated performance for %s: %f.", optimizer_name, performance)
        else:
            logger.warning(
                "No performance_metric provided. Skipping performance update."
            )

    def get_best_performance(self, optimizer_name: str) -> Optional[float]:
        if optimizer_name not in self.performance_history:
            return None
        performances = self.performance_history[optimizer_name]
        if not performances:
            return None
        return min(performances)  # Assuming minimization problem

    def get_average_performance(self, optimizer_name: str) -> Optional[float]:
        if optimizer_name not in self.performance_history:
            return None
        performances = self.performance_history[optimizer_name]
        if not performances:
            return None
        return np.mean(performances)

    def compile(self, space: Space) -> None:
        for optimizer in self.optimizers:
            optimizer.compile(space)
        if self.optimizers:
            self.current_optimizer = self.optimizers[0]
        logger.debug(
            "Compiled hyperheuristic with %d optimizers.", len(self.optimizers)
        )

    def evaluate(self, space: Space, function: Function) -> None:
        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for evaluation")
        self.current_optimizer.evaluate(space, function)
        self.update_performance(self.current_optimizer, space)

    def update(self, space: Space, function: Function = None) -> None:
        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for update")

        # Check if the optimizer needs function as an argument
        import inspect

        sig = inspect.signature(self.current_optimizer.update)
        if "function" in sig.parameters:
            self.current_optimizer.update(space, function)
        else:
            self.current_optimizer.update(space)

        self.iteration += 1
        new_optimizer = self.select_optimizer(space, function)
        if new_optimizer != self.current_optimizer:
            self.current_optimizer = new_optimizer
            logger.debug("Switched to optimizer: %s.", new_optimizer.__class__.__name__)

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_iterations": self.iteration,
            "optimizer_selections": dict(self.selection_count),
            "performance_history": dict(self.performance_history),
            "optimizer_history": list(self.optimizer_history),
        }
        best_performances = {}
        for optimizer_name in self.performance_history:
            best_perf = self.get_best_performance(optimizer_name)
            if best_perf is not None:
                best_performances[optimizer_name] = best_perf
        stats["best_performances"] = best_performances
        avg_performances = {}
        for optimizer_name in self.performance_history:
            avg_perf = self.get_average_performance(optimizer_name)
            if avg_perf is not None:
                avg_performances[optimizer_name] = avg_perf
        stats["average_performances"] = avg_performances
        return stats
