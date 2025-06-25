"""HyperHeuristic.
"""

import copy
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.optimizer import MultiObjectiveOptimizer, Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class HyperHeuristic(Optimizer):
    """A HyperHeuristic class that manages multiple low-level optimizers
    and provides high-level strategies for algorithm selection and adaptation.

    """

    def __init__(self, optimizers: Optional[List[Optimizer]] = None) -> None:
        """Initialization method.

        Args:
            optimizers: List of low-level optimizers to be managed.

        """

        logger.info("Overriding class: Optimizer -> HyperHeuristic.")

        super().__init__()

        self.optimizers = optimizers or []
        self.current_optimizer = None
        self.optimizer_history = []
        self.performance_history = {}
        self.selection_count = {}
        self.iteration = 0

        # Initialize performance tracking for each optimizer
        if self.optimizers:
            for optimizer in self.optimizers:
                optimizer_name = optimizer.__class__.__name__
                self.performance_history[optimizer_name] = []
                self.selection_count[optimizer_name] = 0

        logger.info("Class overrided.")

    @property
    def optimizers(self) -> List[Optimizer]:
        """List of low-level optimizers."""

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
        """Currently selected optimizer."""

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
        """History of selected optimizers."""

        return self._optimizer_history

    @optimizer_history.setter
    def optimizer_history(self, optimizer_history: List[str]) -> None:
        if not isinstance(optimizer_history, list):
            raise e.TypeError("`optimizer_history` should be a list")

        self._optimizer_history = optimizer_history

    @property
    def performance_history(self) -> Dict[str, List[float]]:
        """Performance history for each optimizer."""

        return self._performance_history

    @performance_history.setter
    def performance_history(self, performance_history: Dict[str, List[float]]) -> None:
        if not isinstance(performance_history, dict):
            raise e.TypeError("`performance_history` should be a dictionary")

        self._performance_history = performance_history

    @property
    def selection_count(self) -> Dict[str, int]:
        """Number of times each optimizer was selected."""

        return self._selection_count

    @selection_count.setter
    def selection_count(self, selection_count: Dict[str, int]) -> None:
        if not isinstance(selection_count, dict):
            raise e.TypeError("`selection_count` should be a dictionary")

        self._selection_count = selection_count

    @property
    def iteration(self) -> int:
        """Current iteration number."""

        return self._iteration

    @iteration.setter
    def iteration(self, iteration: int) -> None:
        if not isinstance(iteration, int):
            raise e.TypeError("`iteration` should be an integer")
        if iteration < 0:
            raise e.ValueError("`iteration` should be >= 0")

        self._iteration = iteration

    def add_optimizer(self, optimizer: Optimizer) -> None:
        """Adds a new optimizer to the pool.

        Args:
            optimizer: Optimizer instance to be added.

        """

        if not isinstance(optimizer, Optimizer):
            raise e.TypeError("`optimizer` should be an Optimizer instance")

        self.optimizers.append(optimizer)
        optimizer_name = optimizer.__class__.__name__
        self.performance_history[optimizer_name] = []
        self.selection_count[optimizer_name] = 0

        logger.debug("Added optimizer: %s.", optimizer_name)

    def remove_optimizer(self, optimizer_name: str) -> None:
        """Removes an optimizer from the pool.

        Args:
            optimizer_name: Name of the optimizer to be removed.

        """

        if not isinstance(optimizer_name, str):
            raise e.TypeError("`optimizer_name` should be a string")

        # Find and remove optimizer
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
        """Selects the next optimizer to be used.

        This method should be overridden by specific hyperheuristic implementations
        to implement different selection strategies.

        Args:
            space: Current search space.
            function: Objective function.

        Returns:
            (Optimizer): Selected optimizer.

        """

        if not self.optimizers:
            raise e.ValueError("No optimizers available for selection")

        # Default strategy: round-robin selection
        selected_index = self.iteration % len(self.optimizers)
        selected_optimizer = self.optimizers[selected_index]

        # Update selection count
        optimizer_name = selected_optimizer.__class__.__name__
        self.selection_count[optimizer_name] += 1

        # Update history
        self.optimizer_history.append(optimizer_name)

        logger.debug("Selected optimizer: %s.", optimizer_name)

        return selected_optimizer

    def update_performance(self, optimizer: Optimizer, performance: float) -> None:
        """Updates the performance history of an optimizer.

        Args:
            optimizer: Optimizer instance.
            performance: Performance value to be recorded.

        """

        if not isinstance(optimizer, Optimizer):
            raise e.TypeError("`optimizer` should be an Optimizer instance")

        optimizer_name = optimizer.__class__.__name__

        if optimizer_name not in self.performance_history:
            self.performance_history[optimizer_name] = []

        self.performance_history[optimizer_name].append(performance)

        logger.debug("Updated performance for %s: %f.", optimizer_name, performance)

    def get_best_performance(self, optimizer_name: str) -> Optional[float]:
        """Gets the best performance achieved by an optimizer.

        Args:
            optimizer_name: Name of the optimizer.

        Returns:
            (Optional[float]): Best performance value or None if not available.

        """

        if optimizer_name not in self.performance_history:
            return None

        performances = self.performance_history[optimizer_name]
        if not performances:
            return None

        return min(performances)  # Assuming minimization problem

    def get_average_performance(self, optimizer_name: str) -> Optional[float]:
        """Gets the average performance of an optimizer.

        Args:
            optimizer_name: Name of the optimizer.

        Returns:
            (Optional[float]): Average performance value or None if not available.

        """

        if optimizer_name not in self.performance_history:
            return None

        performances = self.performance_history[optimizer_name]
        if not performances:
            return None

        return np.mean(performances)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this hyperheuristic.

        Args:
            space: A Space object containing meta-information.

        """

        # Compile all optimizers
        for optimizer in self.optimizers:
            optimizer.compile(space)

        # Select initial optimizer
        if self.optimizers:
            self.current_optimizer = self.optimizers[0]

        logger.debug(
            "Compiled hyperheuristic with %d optimizers.", len(self.optimizers)
        )

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space using the current optimizer.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object serving as an objective function.

        """

        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for evaluation")

        # Use the current optimizer to evaluate
        self.current_optimizer.evaluate(space, function)

        # Record performance
        if hasattr(space, "best_agent") and space.best_agent:
            performance = space.best_agent.fit
            self.update_performance(self.current_optimizer, performance)

    def update(self, space: Space) -> None:
        """Updates the search space and potentially selects a new optimizer.

        Args:
            space: A Space object containing agents and update-related information.

        """

        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for update")

        # Update using current optimizer
        self.current_optimizer.update(space)

        # Increment iteration
        self.iteration += 1

        # Select new optimizer for next iteration (if strategy requires)
        # This can be overridden by specific implementations
        new_optimizer = self.select_optimizer(space, None)
        if new_optimizer != self.current_optimizer:
            self.current_optimizer = new_optimizer
            logger.debug("Switched to optimizer: %s.", new_optimizer.__class__.__name__)

    def get_statistics(self) -> Dict[str, Any]:
        """Gets statistics about the hyperheuristic's performance.

        Returns:
            (Dict[str, Any]): Dictionary containing various statistics.

        """

        stats = {
            "total_iterations": self.iteration,
            "optimizer_selections": dict(self.selection_count),
            "performance_history": dict(self.performance_history),
            "optimizer_history": list(self.optimizer_history),
        }

        # Calculate best performances
        best_performances = {}
        for optimizer_name in self.performance_history:
            best_perf = self.get_best_performance(optimizer_name)
            if best_perf is not None:
                best_performances[optimizer_name] = best_perf

        stats["best_performances"] = best_performances

        # Calculate average performances
        avg_performances = {}
        for optimizer_name in self.performance_history:
            avg_perf = self.get_average_performance(optimizer_name)
            if avg_perf is not None:
                avg_performances[optimizer_name] = avg_perf

        stats["average_performances"] = avg_performances

        return stats


class MultiObjectiveHyperHeuristic(HyperHeuristic):
    """A MultiObjectiveHyperHeuristic class that manages multiple multi-objective
    optimizers and provides strategies for multi-objective optimization.

    """

    def __init__(
        self, optimizers: Optional[List[MultiObjectiveOptimizer]] = None
    ) -> None:
        """Initialization method.

        Args:
            optimizers: List of multi-objective optimizers to be managed.

        """

        logger.info("Overriding class: HyperHeuristic -> MultiObjectiveHyperHeuristic.")

        super().__init__(optimizers)

        # Validate that all optimizers are multi-objective
        for optimizer in self.optimizers:
            if not isinstance(optimizer, MultiObjectiveOptimizer):
                raise e.TypeError(
                    "All optimizers should be MultiObjectiveOptimizer instances"
                )

        logger.info("Class overrided.")

    def update_performance(
        self, optimizer: MultiObjectiveOptimizer, pareto_front: List[Agent]
    ) -> None:
        """Updates the performance history using Pareto front metrics.

        Args:
            optimizer: Multi-objective optimizer instance.
            pareto_front: List of agents representing the Pareto front.

        """

        if not isinstance(optimizer, MultiObjectiveOptimizer):
            raise e.TypeError(
                "`optimizer` should be a MultiObjectiveOptimizer instance"
            )

        # Calculate performance metric (e.g., hypervolume, spread, etc.)
        # This is a simplified version - specific implementations can override
        if pareto_front:
            # Simple metric: average of all objectives
            total_fitness = 0
            count = 0
            for agent in pareto_front:
                if hasattr(agent, "fit") and agent.fit is not None:
                    if isinstance(agent.fit, (list, np.ndarray)):
                        total_fitness += np.mean(agent.fit)
                    else:
                        total_fitness += agent.fit
                    count += 1

            if count > 0:
                performance = total_fitness / count
                super().update_performance(optimizer, performance)

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space using the current multi-objective optimizer.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object serving as an objective function.

        """

        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for evaluation")

        # Use the current optimizer to evaluate
        self.current_optimizer.evaluate(space, function)

        # Record performance using Pareto front
        if hasattr(space, "pareto_front") and space.pareto_front:
            self.update_performance(self.current_optimizer, space.pareto_front)
