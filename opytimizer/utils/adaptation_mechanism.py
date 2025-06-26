"""Adaptation mechanisms for hyperheuristics.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class AdaptationMechanism(ABC):
    """Abstract base class for adaptation mechanisms."""

    def __init__(self) -> None:
        """Initialization method."""
        self.name = self.__class__.__name__

    @abstractmethod
    def adapt(
        self, optimizer: Any, performance_history: List[float], iteration: int
    ) -> Dict[str, Any]:
        """Adapt optimizer parameters based on performance.

        Args:
            optimizer: Optimizer to be adapted.
            performance_history: Performance history of the optimizer.
            iteration: Current iteration number.

        Returns:
            (Dict[str, Any]): Adapted parameters.
        """
        pass


class ParameterAdaptation(AdaptationMechanism):
    """Parameter adaptation mechanism."""

    def __init__(self, adaptation_rate: float = 0.1) -> None:
        """Initialization method.

        Args:
            adaptation_rate: Rate of parameter adaptation.
        """
        super().__init__()
        self.adaptation_rate = adaptation_rate

    def adapt(
        self, optimizer: Any, performance_history: List[float], iteration: int
    ) -> Dict[str, Any]:
        """Adapt optimizer parameters.

        Args:
            optimizer: Optimizer to be adapted.
            performance_history: Performance history of the optimizer.
            iteration: Current iteration number.

        Returns:
            (Dict[str, Any]): Adapted parameters.
        """
        if not performance_history:
            return {}

        # Calculate performance trend
        recent_performance = (
            np.mean(performance_history[-5:])
            if len(performance_history) >= 5
            else performance_history[-1]
        )
        historical_performance = np.mean(performance_history)

        # Determine adaptation direction
        if recent_performance < historical_performance:
            # Performance is improving, increase exploration
            adaptation_factor = 1 + self.adaptation_rate
        else:
            # Performance is degrading, increase exploitation
            adaptation_factor = 1 - self.adaptation_rate

        # Adapt parameters based on optimizer type
        adapted_params = {}

        # Example adaptations for common parameters
        if hasattr(optimizer, "w"):  # PSO inertia weight
            adapted_params["w"] = np.clip(optimizer.w * adaptation_factor, 0.1, 0.9)

        if hasattr(optimizer, "c1"):  # PSO cognitive constant
            adapted_params["c1"] = np.clip(optimizer.c1 * adaptation_factor, 0.5, 2.5)

        if hasattr(optimizer, "c2"):  # PSO social constant
            adapted_params["c2"] = np.clip(optimizer.c2 * adaptation_factor, 0.5, 2.5)

        return adapted_params


class StrategyAdaptation(AdaptationMechanism):
    """Strategy adaptation mechanism."""

    def __init__(self, adaptation_threshold: float = 0.1) -> None:
        """Initialization method.

        Args:
            adaptation_threshold: Threshold for strategy adaptation.
        """
        super().__init__()
        self.adaptation_threshold = adaptation_threshold

    def adapt(
        self, optimizer: Any, performance_history: List[float], iteration: int
    ) -> Dict[str, Any]:
        """Adapt optimization strategy.

        Args:
            optimizer: Optimizer to be adapted.
            performance_history: Performance history of the optimizer.
            iteration: Current iteration number.

        Returns:
            (Dict[str, Any]): Adapted strategy parameters.
        """
        if len(performance_history) < 10:
            return {}

        # Calculate performance stability
        recent_performance = performance_history[-10:]
        performance_variance = np.var(recent_performance)

        # If performance is unstable, adapt strategy
        if performance_variance > self.adaptation_threshold:
            # Increase exploration
            return {"exploration_rate": 0.3, "exploitation_rate": 0.7}
        else:
            # Increase exploitation
            return {"exploration_rate": 0.1, "exploitation_rate": 0.9}


class PopulationAdaptation(AdaptationMechanism):
    """Population adaptation mechanism."""

    def __init__(self, min_population: int = 10, max_population: int = 100) -> None:
        """Initialization method.

        Args:
            min_population: Minimum population size.
            max_population: Maximum population size.
        """
        super().__init__()
        self.min_population = min_population
        self.max_population = max_population

    def adapt(
        self, optimizer: Any, performance_history: List[float], iteration: int
    ) -> Dict[str, Any]:
        """Adapt population size.

        Args:
            optimizer: Optimizer to be adapted.
            performance_history: Performance history of the optimizer.
            iteration: Current iteration number.

        Returns:
            (Dict[str, Any]): Adapted population parameters.
        """
        if not performance_history:
            return {}

        # Calculate performance improvement rate
        if len(performance_history) >= 5:
            recent_improvement = performance_history[-1] - performance_history[-5]
        else:
            recent_improvement = 0

        # Adapt population size based on improvement
        if recent_improvement > 0:
            # Good improvement, increase population for more exploration
            population_change = int(self.max_population * 0.1)
        else:
            # Poor improvement, decrease population for more exploitation
            population_change = -int(self.max_population * 0.05)

        current_population = getattr(optimizer, "n_agents", 50)
        new_population = np.clip(
            current_population + population_change,
            self.min_population,
            self.max_population,
        )

        return {"n_agents": int(new_population)}


class AdaptiveParameterControl(AdaptationMechanism):
    """Adaptive parameter control mechanism.

    References:
        Eiben, A. E., Hinterding, R., & Michalewicz, Z. (1999). Parameter control
        in evolutionary algorithms. IEEE Transactions on Evolutionary Computation.
    """

    def __init__(self, adaptation_window: int = 10) -> None:
        """Initialization method.

        Args:
            adaptation_window: Window size for performance analysis.
        """
        super().__init__()
        self.adaptation_window = adaptation_window

    def adapt(
        self, optimizer: Any, performance_history: List[float], iteration: int
    ) -> Dict[str, Any]:
        """Adapt parameters using adaptive control.

        Args:
            optimizer: Optimizer to be adapted.
            performance_history: Performance history of the optimizer.
            iteration: Current iteration number.

        Returns:
            (Dict[str, Any]): Adapted parameters.
        """
        if len(performance_history) < self.adaptation_window:
            return {}

        # Analyze performance trend
        recent_performance = performance_history[-self.adaptation_window :]
        performance_trend = np.polyfit(
            range(len(recent_performance)), recent_performance, 1
        )[0]

        # Calculate adaptation factor based on trend
        if performance_trend < 0:
            # Performance is improving, maintain current parameters
            adaptation_factor = 1.0
        else:
            # Performance is degrading, increase adaptation
            adaptation_factor = 1.0 + abs(performance_trend) * 0.1

        # Adapt common parameters
        adapted_params = {}

        # PSO parameters
        if hasattr(optimizer, "w"):
            adapted_params["w"] = np.clip(optimizer.w * adaptation_factor, 0.1, 0.9)

        # GA parameters
        if hasattr(optimizer, "crossover_rate"):
            adapted_params["crossover_rate"] = np.clip(
                optimizer.crossover_rate * adaptation_factor, 0.1, 1.0
            )

        if hasattr(optimizer, "mutation_rate"):
            adapted_params["mutation_rate"] = np.clip(
                optimizer.mutation_rate * adaptation_factor, 0.001, 0.1
            )

        return adapted_params


class SelfAdaptiveControl(AdaptationMechanism):
    """Self-adaptive parameter control mechanism."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        """Initialization method.

        Args:
            learning_rate: Learning rate for parameter adaptation.
        """
        super().__init__()
        self.learning_rate = learning_rate

    def adapt(
        self, optimizer: Any, performance_history: List[float], iteration: int
    ) -> Dict[str, Any]:
        """Adapt parameters using self-adaptive control.

        Args:
            optimizer: Optimizer to be adapted.
            performance_history: Performance history of the optimizer.
            iteration: Current iteration number.

        Returns:
            (Dict[str, Any]): Adapted parameters.
        """
        if len(performance_history) < 2:
            return {}

        # Calculate performance gradient
        performance_gradient = performance_history[-1] - performance_history[-2]

        # Adapt parameters based on gradient
        adapted_params = {}

        # Example: adapt learning rate based on performance gradient
        if hasattr(optimizer, "learning_rate"):
            new_learning_rate = (
                optimizer.learning_rate + self.learning_rate * performance_gradient
            )
            adapted_params["learning_rate"] = np.clip(new_learning_rate, 0.001, 0.1)

        return adapted_params
