"""Selection strategies for hyperheuristics.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""

    def __init__(self) -> None:
        """Initialization method."""
        self.name = self.__class__.__name__

    @abstractmethod
    def select(
        self, optimizers: List[Any], performance_history: Dict[str, List[float]]
    ) -> Any:
        """Select an optimizer based on the strategy.

        Args:
            optimizers: List of available optimizers.
            performance_history: Performance history for each optimizer.

        Returns:
            (Any): Selected optimizer.
        """
        pass


class ChoiceFunction(SelectionStrategy):
    """Choice Function selection strategy.

    References:
        A. J. Parkes and D. H. Smith. A Hyper-Heuristic Approach to Scheduling.
        Journal of Scheduling (2003).
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5) -> None:
        """Initialization method.

        Args:
            alpha: Weight for recent performance.
            beta: Weight for historical performance.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def select(
        self, optimizers: List[Any], performance_history: Dict[str, List[float]]
    ) -> Any:
        """Select optimizer using Choice Function.

        Args:
            optimizers: List of available optimizers.
            performance_history: Performance history for each optimizer.

        Returns:
            (Any): Selected optimizer.
        """
        if not optimizers:
            raise e.ValueError("No optimizers available for selection")

        choice_values = []
        for optimizer in optimizers:
            optimizer_name = optimizer.__class__.__name__
            if (
                optimizer_name in performance_history
                and performance_history[optimizer_name]
            ):
                recent_perf = performance_history[optimizer_name][-1]
                historical_perf = np.mean(performance_history[optimizer_name])
                choice_value = self.alpha * recent_perf + self.beta * historical_perf
            else:
                choice_value = 0.0  # Default for new optimizers
            choice_values.append(choice_value)

        # Select optimizer with highest choice value
        selected_index = np.argmax(choice_values)
        return optimizers[selected_index]


class MultiArmedBandit(SelectionStrategy):
    """Multi-Armed Bandit selection strategy.

    References:
        A. Aleti and I. Moser. A Systematic Literature Review of Adaptive
        Parameter Control Methods for Evolutionary Algorithms.
        ACM Computing Surveys (2016).
    """

    def __init__(self, exploration_rate: float = 0.1) -> None:
        """Initialization method.

        Args:
            exploration_rate: Rate of exploration vs exploitation.
        """
        super().__init__()
        self.exploration_rate = exploration_rate

    def select(
        self, optimizers: List[Any], performance_history: Dict[str, List[float]]
    ) -> Any:
        """Select optimizer using Multi-Armed Bandit.

        Args:
            optimizers: List of available optimizers.
            performance_history: Performance history for each optimizer.

        Returns:
            (Any): Selected optimizer.
        """
        if not optimizers:
            raise e.ValueError("No optimizers available for selection")

        # Exploration: select random optimizer
        if random.random() < self.exploration_rate:
            return random.choice(optimizers)

        # Exploitation: select best performing optimizer
        best_performance = float("-inf")
        best_optimizer = optimizers[0]

        for optimizer in optimizers:
            optimizer_name = optimizer.__class__.__name__
            if (
                optimizer_name in performance_history
                and performance_history[optimizer_name]
            ):
                avg_performance = np.mean(performance_history[optimizer_name])
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_optimizer = optimizer

        return best_optimizer


class RandomDescent(SelectionStrategy):
    """Random Descent selection strategy.

    References:
        P. Cowling, G. Kendall, and E. Soubeiga. A Hyperheuristic Approach to
        Scheduling a Sales Summit. Practice and Theory of Automated Timetabling (2001).
    """

    def __init__(self, acceptance_threshold: float = 0.1) -> None:
        """Initialization method.

        Args:
            acceptance_threshold: Threshold for accepting worse performance.
        """
        super().__init__()
        self.acceptance_threshold = acceptance_threshold
        self.last_performance = None

    def select(
        self, optimizers: List[Any], performance_history: Dict[str, List[float]]
    ) -> Any:
        """Select optimizer using Random Descent.

        Args:
            optimizers: List of available optimizers.
            performance_history: Performance history for each optimizer.

        Returns:
            (Any): Selected optimizer.
        """
        if not optimizers:
            raise e.ValueError("No optimizers available for selection")

        # Always select random optimizer
        selected_optimizer = random.choice(optimizers)

        # Check if we should accept worse performance
        if self.last_performance is not None:
            current_performance = self._get_current_performance(performance_history)
            if current_performance > self.last_performance:
                if random.random() > self.acceptance_threshold:
                    # Reject and keep previous optimizer
                    return self._get_previous_optimizer(optimizers, performance_history)

        self.last_performance = self._get_current_performance(performance_history)
        return selected_optimizer

    def _get_current_performance(
        self, performance_history: Dict[str, List[float]]
    ) -> float:
        """Get current overall performance."""
        if not performance_history:
            return 0.0
        all_performances = []
        for performances in performance_history.values():
            if performances:
                all_performances.extend(performances)
        return np.mean(all_performances) if all_performances else 0.0

    def _get_previous_optimizer(
        self, optimizers: List[Any], performance_history: Dict[str, List[float]]
    ) -> Any:
        """Get the previously selected optimizer."""
        # Simple implementation: return the first optimizer
        # In a real implementation, you might want to track the previous selection
        return optimizers[0]


class GreedySelection(SelectionStrategy):
    """Greedy selection strategy - always selects the best performing optimizer."""

    def __init__(self) -> None:
        """Initialization method."""
        super().__init__()

    def select(
        self, optimizers: List[Any], performance_history: Dict[str, List[float]]
    ) -> Any:
        """Select optimizer using Greedy strategy.

        Args:
            optimizers: List of available optimizers.
            performance_history: Performance history for each optimizer.

        Returns:
            (Any): Selected optimizer.
        """
        if not optimizers:
            raise e.ValueError("No optimizers available for selection")

        best_performance = float("-inf")
        best_optimizer = optimizers[0]

        for optimizer in optimizers:
            optimizer_name = optimizer.__class__.__name__
            if (
                optimizer_name in performance_history
                and performance_history[optimizer_name]
            ):
                # Use the best performance achieved by this optimizer
                best_opt_performance = min(performance_history[optimizer_name])
                if best_opt_performance > best_performance:
                    best_performance = best_opt_performance
                    best_optimizer = optimizer

        return best_optimizer


class SimulatedAnnealingSelection(SelectionStrategy):
    """Simulated Annealing selection strategy."""

    def __init__(
        self, initial_temperature: float = 100.0, cooling_rate: float = 0.95
    ) -> None:
        """Initialization method.

        Args:
            initial_temperature: Initial temperature for SA.
            cooling_rate: Rate at which temperature decreases.
        """
        super().__init__()
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.temperature = initial_temperature
        self.iteration = 0

    def select(
        self, optimizers: List[Any], performance_history: Dict[str, List[float]]
    ) -> Any:
        """Select optimizer using Simulated Annealing.

        Args:
            optimizers: List of available optimizers.
            performance_history: Performance history for each optimizer.

        Returns:
            (Any): Selected optimizer.
        """
        if not optimizers:
            raise e.ValueError("No optimizers available for selection")

        # Update temperature
        self.temperature *= self.cooling_rate
        self.iteration += 1

        # Calculate acceptance probability
        acceptance_prob = np.exp(-1.0 / self.temperature)

        # With some probability, select random optimizer (exploration)
        if random.random() < acceptance_prob:
            return random.choice(optimizers)

        # Otherwise, select best optimizer (exploitation)
        best_performance = float("-inf")
        best_optimizer = optimizers[0]

        for optimizer in optimizers:
            optimizer_name = optimizer.__class__.__name__
            if (
                optimizer_name in performance_history
                and performance_history[optimizer_name]
            ):
                avg_performance = np.mean(performance_history[optimizer_name])
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_optimizer = optimizer

        return best_optimizer

    def reset(self) -> None:
        """Reset temperature and iteration counter."""
        self.temperature = self.initial_temperature
        self.iteration = 0
