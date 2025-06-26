"""Selection-based hyperheuristic implementation.
"""

from typing import Any, Dict, List, Optional

import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.hyperheuristic import HyperHeuristic
from opytimizer.core.space import Space
from opytimizer.utils import logging
from opytimizer.utils.selection_strategy import ChoiceFunction, SelectionStrategy

logger = logging.get_logger(__name__)


class SelectionHyperHeuristic(HyperHeuristic):
    """A selection-based hyperheuristic that uses different strategies
    to select between low-level optimizers.

    This hyperheuristic implements the selection approach where different
    low-level optimizers are selected based on their performance history
    and the chosen selection strategy.

    References:
        E. K. Burke et al. Hyper-heuristics: A survey of the state of the art.
        Journal of the Operational Research Society (2013).
    """

    def __init__(
        self,
        optimizers: Optional[List[Any]] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
        performance_metric=None,
        selection_interval: int = 1,
    ) -> None:
        """Initialization method.

        Args:
            optimizers: List of low-level optimizers to be managed.
            selection_strategy: Strategy for selecting optimizers.
            performance_metric: Function to evaluate optimizer performance.
            selection_interval: Interval between optimizer selections.
        """
        super().__init__(optimizers, performance_metric)

        self.selection_strategy = selection_strategy or ChoiceFunction()
        self.selection_interval = selection_interval
        self.last_selection_iteration = 0

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

    def select_optimizer(self, space: Space, function: Function) -> Any:
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

    def update(self, space: Space) -> None:
        """Update the search space and potentially select a new optimizer.

        Args:
            space: A Space object containing agents and update-related information.
        """
        if not self.current_optimizer:
            raise e.ValueError("No optimizer selected for update")

        # Update using current optimizer
        self.current_optimizer.update(space)

        # Increment iteration
        self.iteration += 1

        # Select new optimizer if needed
        new_optimizer = self.select_optimizer(space, None)
        if new_optimizer != self.current_optimizer:
            self.current_optimizer = new_optimizer
            logger.debug("Switched to optimizer: %s.", new_optimizer.__class__.__name__)

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about the selection strategy.

        Returns:
            (Dict[str, Any]): Dictionary containing strategy statistics.
        """
        stats = super().get_statistics()
        stats.update(
            {
                "selection_strategy": self.selection_strategy.name,
                "selection_interval": self.selection_interval,
                "last_selection_iteration": self.last_selection_iteration,
            }
        )
        return stats
