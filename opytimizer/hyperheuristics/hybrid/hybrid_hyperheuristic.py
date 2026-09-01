"""Hybrid hyperheuristic implementation.
"""

from typing import Any, Dict, List, Optional

import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.hyperheuristic import HyperHeuristic
from opytimizer.core.space import _Space
from opytimizer.hyperheuristics.adaptation_mechanism import (
    AdaptationMechanism,
    ParameterAdaptation,
)
from opytimizer.hyperheuristics.selection_strategy import (
    ChoiceFunction,
    SelectionStrategy,
)
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class HybridHyperHeuristic(HyperHeuristic):
    """A hybrid hyperheuristic that combines selection and adaptation approaches.

    This hyperheuristic implements a hybrid approach where both optimizer
    selection and parameter adaptation are used together to improve performance.
    """

    def __init__(
        self,
        optimizers: Optional[List[Any]] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
        adaptation_mechanism: Optional[AdaptationMechanism] = None,
        performance_metric=None,
        selection_interval: int = 1,
        adaptation_interval: int = 5,
    ) -> None:
        """Initialization method.

        Args:
            optimizers: List of low-level optimizers to be managed.
            selection_strategy: Strategy for selecting optimizers.
            adaptation_mechanism: Mechanism for adapting optimizer parameters.
            performance_metric: Function to evaluate optimizer performance.
            selection_interval: Interval between optimizer selections.
            adaptation_interval: Interval between parameter adaptations.
        """
        super().__init__(optimizers, performance_metric)

        self.selection_strategy = selection_strategy or ChoiceFunction()
        self.adaptation_mechanism = adaptation_mechanism or ParameterAdaptation()
        self.selection_interval = selection_interval
        self.adaptation_interval = adaptation_interval
        self.last_selection_iteration = 0
        self.last_adaptation_iteration = 0
        self.adaptation_history = []

    @property
    def selection_strategy(self) -> SelectionStrategy:
        """Selection strategy used by the hyperheuristic."""
        return self._selection_strategy

    @selection_strategy.setter
    def selection_strategy(self, selection_strategy: SelectionStrategy) -> None:
        if not isinstance(selection_strategy, SelectionStrategy):
            raise e.TypeError(
                "`selection_strategy` should be a SelectionStrategy instance"
            )
        self._selection_strategy = selection_strategy

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
    def selection_interval(self) -> int:
        """Interval between optimizer selections."""
        return self._selection_interval

    @selection_interval.setter
    def selection_interval(self, selection_interval: int) -> None:
        if not isinstance(selection_interval, int):
            raise e.TypeError("`selection_interval` should be an integer")
        if selection_interval < 1:
            raise e.ValueError("`selection_interval` should be >= 1")
        self._selection_interval = selection_interval

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

    def select_optimizer(self, space: _Space, function: Function) -> Any:
        """Select optimizer using the configured selection strategy.

        Args:
            space: Current search space.
            function: Objective function.

        Returns:
            (Any): Selected optimizer.
        """
        # Check if it's time to select a new optimizer
        if (self.iteration - self.last_selection_iteration) >= self.selection_interval:
            selected_optimizer = self.selection_strategy.select(
                self.optimizers, self.performance_history
            )

            # Update selection tracking
            optimizer_name = selected_optimizer.__class__.__name__
            self.selection_count[optimizer_name] += 1
            self.optimizer_history.append(optimizer_name)
            self.last_selection_iteration = self.iteration

            logger.debug(
                "Selected optimizer: %s using strategy: %s",
                optimizer_name,
                self.selection_strategy.name,
            )

            return selected_optimizer

        # Return current optimizer if not time to select
        return self.current_optimizer

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

    def update(self, space: _Space, function: Function = None) -> None:
        """Update the search space, potentially select a new optimizer, and adapt parameters.

        Args:
            space: A Space object containing agents and update-related information.
            function: Objective function (optional, for optimizers that need it).
        """
        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for update")

        # Check if the optimizer needs function as an argument
        import inspect

        sig = inspect.signature(self.current_optimizer.update)
        if "function" in sig.parameters:
            self.current_optimizer.update(space, function)
        else:
            self.current_optimizer.update(space)

        # Increment iteration
        self.iteration += 1

        # Check if it's time to select a new optimizer
        new_optimizer = self.select_optimizer(space, function)
        if new_optimizer != self.current_optimizer:
            self.current_optimizer = new_optimizer
            logger.debug("Switched to optimizer: %s.", new_optimizer.__class__.__name__)

        # Check if it's time to adapt parameters
        if (
            self.iteration - self.last_adaptation_iteration
        ) >= self.adaptation_interval:
            self.adapt_optimizer(self.current_optimizer)
            self.last_adaptation_iteration = self.iteration

    def get_hybrid_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hybrid hyperheuristic.

        Returns:
            (Dict[str, Any]): Dictionary containing hybrid statistics.
        """
        stats = super().get_statistics()
        stats.update(
            {
                "selection_strategy": self.selection_strategy.name,
                "adaptation_mechanism": self.adaptation_mechanism.name,
                "selection_interval": self.selection_interval,
                "adaptation_interval": self.adaptation_interval,
                "last_selection_iteration": self.last_selection_iteration,
                "last_adaptation_iteration": self.last_adaptation_iteration,
                "adaptation_history": self.adaptation_history,
            }
        )
        return stats
