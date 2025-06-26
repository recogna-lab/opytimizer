"""Adaptive hyperheuristic implementation.
"""

from typing import Any, Dict, List, Optional

import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.hyperheuristic import HyperHeuristic
from opytimizer.core.space import Space
from opytimizer.utils import logging
from opytimizer.utils.adaptation_mechanism import (
    AdaptationMechanism,
    ParameterAdaptation,
)

logger = logging.get_logger(__name__)


class AdaptiveHyperHeuristic(HyperHeuristic):
    """An adaptive hyperheuristic that adapts optimizer parameters
    based on performance feedback.

    This hyperheuristic implements the adaptive approach where optimizer
    parameters are modified dynamically based on their performance history
    and the chosen adaptation mechanism.

    References:
        A. E. Eiben and J. E. Smith. Introduction to Evolutionary Computing.
        Springer (2015).
    """

    def __init__(
        self,
        optimizers: Optional[List[Any]] = None,
        adaptation_mechanism: Optional[AdaptationMechanism] = None,
        performance_metric=None,
        adaptation_interval: int = 5,
    ) -> None:
        """Initialization method.

        Args:
            optimizers: List of low-level optimizers to be managed.
            adaptation_mechanism: Mechanism for adapting optimizer parameters.
            performance_metric: Function to evaluate optimizer performance.
            adaptation_interval: Interval between parameter adaptations.
        """
        super().__init__(optimizers, performance_metric)

        self.adaptation_mechanism = adaptation_mechanism or ParameterAdaptation()
        self.adaptation_interval = adaptation_interval
        self.last_adaptation_iteration = 0
        self.adaptation_history = []

    @property
    def adaptation_mechanism(self) -> AdaptationMechanism:
        """Adaptation mechanism used by the hyperheuristic."""
        return self._adaptation_mechanism

    @adaptation_mechanism.setter
    def adaptation_mechanism(self, adaptation_mechanism: AdaptationMechanism) -> None:
        if not isinstance(adaptation_mechanism, AdaptationMechanism):
            raise e.TypeError(
                "`adaptation_mechanism` should be an AdaptationMechanism instance"
            )
        self._adaptation_mechanism = adaptation_mechanism

    @property
    def adaptation_interval(self) -> int:
        """Interval between parameter adaptations."""
        return self._adaptation_interval

    @adaptation_interval.setter
    def adaptation_interval(self, adaptation_interval: int) -> None:
        if not isinstance(adaptation_interval, int):
            raise e.TypeError("`adaptation_interval` should be an integer")
        if adaptation_interval < 1:
            raise e.ValueError("`adaptation_interval` should be >= 1")
        self._adaptation_interval = adaptation_interval

    def adapt_optimizer(self, optimizer: Any) -> None:
        """Adapt the parameters of an optimizer.

        Args:
            optimizer: Optimizer to be adapted.
        """
        optimizer_name = optimizer.__class__.__name__

        if optimizer_name in self.performance_history:
            performance_history = self.performance_history[optimizer_name]

            # Get adapted parameters
            adapted_params = self.adaptation_mechanism.adapt(
                optimizer, performance_history, self.iteration
            )

            # Apply adapted parameters
            for param_name, param_value in adapted_params.items():
                if hasattr(optimizer, param_name):
                    setattr(optimizer, param_name, param_value)
                    logger.debug(
                        "Adapted parameter %s for %s: %f",
                        param_name,
                        optimizer_name,
                        param_value,
                    )

            # Record adaptation
            self.adaptation_history.append(
                {
                    "iteration": self.iteration,
                    "optimizer": optimizer_name,
                    "adapted_params": adapted_params,
                }
            )

    def update(self, space: Space) -> None:
        """Update the search space and potentially adapt optimizer parameters.

        Args:
            space: A Space object containing agents and update-related information.
        """
        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for update")

        # Update using current optimizer
        self.current_optimizer.update(space)

        # Increment iteration
        self.iteration += 1

        # Check if it's time to adapt parameters
        if (
            self.iteration - self.last_adaptation_iteration
        ) >= self.adaptation_interval:
            self.adapt_optimizer(self.current_optimizer)
            self.last_adaptation_iteration = self.iteration

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the adaptation mechanism.

        Returns:
            (Dict[str, Any]): Dictionary containing adaptation statistics.
        """
        stats = super().get_statistics()
        stats.update(
            {
                "adaptation_mechanism": self.adaptation_mechanism.name,
                "adaptation_interval": self.adaptation_interval,
                "last_adaptation_iteration": self.last_adaptation_iteration,
                "adaptation_history": self.adaptation_history,
            }
        )
        return stats
