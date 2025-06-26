"""Performance tracking utilities for hyperheuristics.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class PerformanceTracker:
    """Performance tracker for hyperheuristics."""

    def __init__(self, window_size: int = 10) -> None:
        """Initialization method.

        Args:
            window_size: Size of the performance window for analysis.
        """
        self.window_size = window_size
        self.performance_history = {}
        self.selection_history = {}
        self.timing_history = {}
        self.iteration_history = []

    def update_performance(
        self, optimizer_name: str, performance: float, iteration: int
    ) -> None:
        """Update performance for an optimizer.

        Args:
            optimizer_name: Name of the optimizer.
            performance: Performance value.
            iteration: Current iteration.
        """
        if optimizer_name not in self.performance_history:
            self.performance_history[optimizer_name] = []

        self.performance_history[optimizer_name].append(performance)

        # Keep only the last window_size entries
        if len(self.performance_history[optimizer_name]) > self.window_size:
            self.performance_history[optimizer_name] = self.performance_history[
                optimizer_name
            ][-self.window_size :]

    def update_selection(self, optimizer_name: str, iteration: int) -> None:
        """Update selection history for an optimizer.

        Args:
            optimizer_name: Name of the optimizer.
            iteration: Current iteration.
        """
        if optimizer_name not in self.selection_history:
            self.selection_history[optimizer_name] = []

        self.selection_history[optimizer_name].append(iteration)

    def update_timing(self, optimizer_name: str, execution_time: float) -> None:
        """Update timing information for an optimizer.

        Args:
            optimizer_name: Name of the optimizer.
            execution_time: Execution time in seconds.
        """
        if optimizer_name not in self.timing_history:
            self.timing_history[optimizer_name] = []

        self.timing_history[optimizer_name].append(execution_time)

    def get_best_performance(self, optimizer_name: str) -> Optional[float]:
        """Get the best performance achieved by an optimizer.

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
        """Get the average performance of an optimizer.

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

    def get_performance_trend(self, optimizer_name: str) -> Optional[float]:
        """Get the performance trend of an optimizer.

        Args:
            optimizer_name: Name of the optimizer.

        Returns:
            (Optional[float]): Performance trend (slope) or None if not available.
        """
        if optimizer_name not in self.performance_history:
            return None

        performances = self.performance_history[optimizer_name]
        if len(performances) < 2:
            return None

        # Calculate linear trend
        x = np.arange(len(performances))
        slope = np.polyfit(x, performances, 1)[0]
        return slope

    def get_selection_frequency(self, optimizer_name: str) -> float:
        """Get the selection frequency of an optimizer.

        Args:
            optimizer_name: Name of the optimizer.

        Returns:
            (float): Selection frequency (0.0 to 1.0).
        """
        if optimizer_name not in self.selection_history:
            return 0.0

        total_selections = len(self.selection_history[optimizer_name])
        total_iterations = len(self.iteration_history)

        if total_iterations == 0:
            return 0.0

        return total_selections / total_iterations

    def get_average_execution_time(self, optimizer_name: str) -> Optional[float]:
        """Get the average execution time of an optimizer.

        Args:
            optimizer_name: Name of the optimizer.

        Returns:
            (Optional[float]): Average execution time in seconds or None if not available.
        """
        if optimizer_name not in self.timing_history:
            return None

        execution_times = self.timing_history[optimizer_name]
        if not execution_times:
            return None

        return np.mean(execution_times)

    def get_performance_ranking(self) -> List[Tuple[str, float]]:
        """Get performance ranking of all optimizers.

        Returns:
            (List[Tuple[str, float]]): List of (optimizer_name, best_performance) tuples.
        """
        rankings = []

        for optimizer_name in self.performance_history:
            best_performance = self.get_best_performance(optimizer_name)
            if best_performance is not None:
                rankings.append((optimizer_name, best_performance))

        # Sort by performance (ascending for minimization)
        rankings.sort(key=lambda x: x[1])
        return rankings

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all optimizers.

        Returns:
            (Dict[str, Any]): Dictionary containing various statistics.
        """
        stats = {
            "performance_history": dict(self.performance_history),
            "selection_history": dict(self.selection_history),
            "timing_history": dict(self.timing_history),
            "performance_ranking": self.get_performance_ranking(),
        }

        # Add individual optimizer statistics
        optimizer_stats = {}
        for optimizer_name in self.performance_history:
            optimizer_stats[optimizer_name] = {
                "best_performance": self.get_best_performance(optimizer_name),
                "average_performance": self.get_average_performance(optimizer_name),
                "performance_trend": self.get_performance_trend(optimizer_name),
                "selection_frequency": self.get_selection_frequency(optimizer_name),
                "average_execution_time": self.get_average_execution_time(
                    optimizer_name
                ),
            }

        stats["optimizer_statistics"] = optimizer_stats
        return stats

    def reset(self) -> None:
        """Reset all tracking data."""
        self.performance_history.clear()
        self.selection_history.clear()
        self.timing_history.clear()
        self.iteration_history.clear()


class MultiObjectivePerformanceTracker(PerformanceTracker):
    """Performance tracker for multi-objective optimization."""

    def __init__(self, window_size: int = 10) -> None:
        """Initialization method.

        Args:
            window_size: Size of the performance window for analysis.
        """
        super().__init__(window_size)
        self.hypervolume_history = {}
        self.spread_history = {}
        self.epsilon_history = {}

    def update_hypervolume(self, optimizer_name: str, hypervolume: float) -> None:
        """Update hypervolume for an optimizer.

        Args:
            optimizer_name: Name of the optimizer.
            hypervolume: Hypervolume value.
        """
        if optimizer_name not in self.hypervolume_history:
            self.hypervolume_history[optimizer_name] = []

        self.hypervolume_history[optimizer_name].append(hypervolume)

        # Keep only the last window_size entries
        if len(self.hypervolume_history[optimizer_name]) > self.window_size:
            self.hypervolume_history[optimizer_name] = self.hypervolume_history[
                optimizer_name
            ][-self.window_size :]

    def update_spread(self, optimizer_name: str, spread: float) -> None:
        """Update spread for an optimizer.

        Args:
            optimizer_name: Name of the optimizer.
            spread: Spread value.
        """
        if optimizer_name not in self.spread_history:
            self.spread_history[optimizer_name] = []

        self.spread_history[optimizer_name].append(spread)

        # Keep only the last window_size entries
        if len(self.spread_history[optimizer_name]) > self.window_size:
            self.spread_history[optimizer_name] = self.spread_history[optimizer_name][
                -self.window_size :
            ]

    def update_epsilon(self, optimizer_name: str, epsilon: float) -> None:
        """Update epsilon indicator for an optimizer.

        Args:
            optimizer_name: Name of the optimizer.
            epsilon: Epsilon indicator value.
        """
        if optimizer_name not in self.epsilon_history:
            self.epsilon_history[optimizer_name] = []

        self.epsilon_history[optimizer_name].append(epsilon)

        # Keep only the last window_size entries
        if len(self.epsilon_history[optimizer_name]) > self.window_size:
            self.epsilon_history[optimizer_name] = self.epsilon_history[optimizer_name][
                -self.window_size :
            ]

    def get_best_hypervolume(self, optimizer_name: str) -> Optional[float]:
        """Get the best hypervolume achieved by an optimizer.

        Args:
            optimizer_name: Name of the optimizer.

        Returns:
            (Optional[float]): Best hypervolume value or None if not available.
        """
        if optimizer_name not in self.hypervolume_history:
            return None

        hypervolumes = self.hypervolume_history[optimizer_name]
        if not hypervolumes:
            return None

        return max(hypervolumes)  # Higher hypervolume is better

    def get_average_hypervolume(self, optimizer_name: str) -> Optional[float]:
        """Get the average hypervolume of an optimizer.

        Args:
            optimizer_name: Name of the optimizer.

        Returns:
            (Optional[float]): Average hypervolume value or None if not available.
        """
        if optimizer_name not in self.hypervolume_history:
            return None

        hypervolumes = self.hypervolume_history[optimizer_name]
        if not hypervolumes:
            return None

        return np.mean(hypervolumes)

    def get_hypervolume_ranking(self) -> List[Tuple[str, float]]:
        """Get hypervolume ranking of all optimizers.

        Returns:
            (List[Tuple[str, float]]): List of (optimizer_name, best_hypervolume) tuples.
        """
        rankings = []

        for optimizer_name in self.hypervolume_history:
            best_hypervolume = self.get_best_hypervolume(optimizer_name)
            if best_hypervolume is not None:
                rankings.append((optimizer_name, best_hypervolume))

        # Sort by hypervolume (descending for maximization)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for multi-objective optimization.

        Returns:
            (Dict[str, Any]): Dictionary containing various statistics.
        """
        stats = super().get_statistics()

        # Add multi-objective specific statistics
        stats["hypervolume_history"] = dict(self.hypervolume_history)
        stats["spread_history"] = dict(self.spread_history)
        stats["epsilon_history"] = dict(self.epsilon_history)
        stats["hypervolume_ranking"] = self.get_hypervolume_ranking()

        # Update optimizer statistics with multi-objective metrics
        for optimizer_name in self.hypervolume_history:
            if optimizer_name in stats["optimizer_statistics"]:
                stats["optimizer_statistics"][optimizer_name].update(
                    {
                        "best_hypervolume": self.get_best_hypervolume(optimizer_name),
                        "average_hypervolume": self.get_average_hypervolume(
                            optimizer_name
                        ),
                    }
                )

        return stats

    def reset(self) -> None:
        """Reset all tracking data."""
        super().reset()
        self.hypervolume_history.clear()
        self.spread_history.clear()
        self.epsilon_history.clear()
