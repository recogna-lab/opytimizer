"""Callbacks.
"""

from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.space import _SingleObjectiveSpace, _MultiObjectiveSpace

Opytimizer = TypeVar("Opytimizer")


class Callback:
    """A Callback class that handles additional variables and methods
    manipulation that are not provided by the library.

    """

    def __init__(self):
        """Initialization method."""

        pass

    def on_task_begin(self, opt_model: Opytimizer) -> None:
        """Performs a callback whenever a task begins.

        Args:
            opt_model: An instance of the optimization model.

        """

        pass

    def on_task_end(self, opt_model: Opytimizer) -> None:
        """Performs a callback whenever a task ends.

        Args:
            opt_model: An instance of the optimization model.

        """

        pass

    def on_iteration_begin(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a callback whenever an iteration begins.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_iteration_end(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a callback whenever an iteration ends.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_evaluate_before(self, *evaluate_args) -> None:
        """Performs a callback prior to the `evaluate` method."""

        pass

    def on_evaluate_after(self, *evaluate_args) -> None:
        """Performs a callback after the `evaluate` method."""

        pass

    def on_update_before(self, *update_args) -> None:
        """Performs a callback prior to the `update` method."""

        pass

    def on_update_after(self, *update_args) -> None:
        """Performs a callback after the `update` method."""

        pass


class CallbackVessel:
    """Wraps multiple callbacks in an ready-to-use class."""

    def __init__(self, callbacks: List[Callback]) -> None:
        """Initialization method.

        Args:
            callbacks: List of Callback-based childs.

        """

        self.callbacks = callbacks or []

    @property
    def callbacks(self) -> List[Callback]:
        """List of Callback-based childs."""

        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: List[Callback]) -> None:
        if not isinstance(callbacks, list):
            raise e.TypeError("`callbacks` should be a list")

        self._callbacks = callbacks

    def on_task_begin(self, opt_model: Opytimizer) -> None:
        """Performs a list of callbacks whenever a task begins.

        Args:
            opt_model: An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_task_begin(opt_model)

    def on_task_end(self, opt_model: Opytimizer) -> None:
        """Performs a list of callbacks whenever a task ends.

        Args:
            opt_model: An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_task_end(opt_model)

    def on_iteration_begin(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a list of callbacks whenever an iteration begins.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_iteration_begin(iteration, opt_model)

    def on_iteration_end(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a list of callbacks whenever an iteration ends.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_iteration_end(iteration, opt_model)

    def on_evaluate_before(self, *evaluate_args) -> None:
        """Performs a list of callbacks prior to the `evaluate` method."""

        for callback in self.callbacks:
            callback.on_evaluate_before(*evaluate_args)

    def on_evaluate_after(self, *evaluate_args) -> None:
        """Performs a list of callbacks after the `evaluate` method."""

        for callback in self.callbacks:
            callback.on_evaluate_after(*evaluate_args)

    def on_update_before(self, *update_args) -> None:
        """Performs a list of callbacks prior to the `update` method."""

        for callback in self.callbacks:
            callback.on_update_before(*update_args)

    def on_update_after(self, *update_args) -> None:
        """Performs a list of callbacks after the `update` method."""

        for callback in self.callbacks:
            callback.on_update_after(*update_args)


class CheckpointCallback(Callback):
    """A CheckpointCallback class that handles additional logging and
    model's checkpointing.

    """

    def __init__(self, file_path: str = None, frequency: int = 0) -> None:
        """Initialization method.

        Args:
            file_path: Path of file to be saved.
            frequency: Interval between checkpoints.

        """

        super(CheckpointCallback, self).__init__()

        self.file_path = file_path or "checkpoint.pkl"
        self.frequency = frequency

    @property
    def file_path(self) -> str:
        """File's path."""

        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str) -> None:
        if not isinstance(file_path, str):
            raise e.TypeError("`file_path` should be a string")

        self._file_path = file_path

    @property
    def frequency(self) -> int:
        """Interval between checkpoints."""

        return self._frequency

    @frequency.setter
    def frequency(self, frequency: int) -> None:
        if not isinstance(frequency, int):
            raise e.TypeError("`frequency` should be an integer")
        if frequency < 0:
            raise e.ValueError("`frequency` should be >= 0")

        self._frequency = frequency

    def on_iteration_end(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a callback whenever an iteration ends.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        if self.frequency > 0:
            if iteration % self.frequency == 0:
                opt_model.save(f"iter_{iteration}_{self.file_path}")


class DiscreteSearchCallback(Callback):
    """A DiscreteSearchCallback class that handles mapping floating-point variables
    to discrete values.

    """

    def __init__(self, allowed_values: List[Union[int, float]] = None) -> None:
        """Initialization method.

        Args:
            allowed_values: Possible values between lower and upper bounds that variables can be mapped.

        """

        super(DiscreteSearchCallback, self).__init__()

        if allowed_values is not None:
            self.allowed_values = allowed_values
        else:
            self.allowed_values = []

    @property
    def allowed_values(self) -> List[Union[int, float]]:
        """Allowed values between lower and upper bounds."""

        return self._allowed_values

    @allowed_values.setter
    def allowed_values(self, allowed_values: List[Union[int, float]]) -> None:
        if not isinstance(allowed_values, list):
            raise e.TypeError("`allowed_values` should be a list")

        self._allowed_values = allowed_values

    def on_task_begin(self, opt_model: Opytimizer) -> None:
        """Performs a callback whenever a task begins.

        Args:
            opt_model: An instance of the optimization model.

        """

        n_variables = opt_model.space.n_variables
        lower_bound = opt_model.space.lb
        upper_bound = opt_model.space.ub

        assert (
            len(self.allowed_values) == n_variables
        ), f"`allowed_values` should have length equals to {n_variables}."
        assert np.all(
            [
                np.all((av >= lb) == (av <= ub))
                for av, lb, ub in zip(self.allowed_values, lower_bound, upper_bound)
            ]
        ), "Every value from `allowed_values` should be between `lower_bound` and `upper_bound`."

    def on_evaluate_before(self, *evaluate_args) -> None:
        """Performs a callback prior to the `evaluate` method."""

        space = evaluate_args[0]
        assert isinstance(
            space, (_SingleObjectiveSpace, _MultiObjectiveSpace)
        ), "`evaluate_args[0]` is not derived from Space class."

        for agent in space.agents:
            for i in range(agent.n_variables):
                # Gathers the current closest allowed value and replaces agent's value
                min_value_idx = np.argmin(
                    abs(agent.position[i] - self.allowed_values[i])
                )
                agent.position[i] = self.allowed_values[i][min_value_idx]


class PerformanceTrackingCallback(Callback):
    """A PerformanceTrackingCallback class that tracks optimizer performance
    during hyperheuristic optimization.

    """

    def __init__(self, window_size: int = 10) -> None:
        """Initialization method.

        Args:
            window_size: Size of the performance window for analysis.
        """
        super(PerformanceTrackingCallback, self).__init__()

        self.window_size = window_size
        self.performance_history = {}
        self.selection_history = {}
        self.timing_history = {}
        self.iteration_history = []
        self.current_optimizer = None

    @property
    def window_size(self) -> int:
        """Size of the performance window."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int) -> None:
        if not isinstance(window_size, int):
            raise e.TypeError("`window_size` should be an integer")
        if window_size < 1:
            raise e.ValueError("`window_size` should be >= 1")
        self._window_size = window_size

    def on_task_begin(self, opt_model: Opytimizer) -> None:
        """Initialize tracking when optimization begins."""
        # Reset tracking data
        self.performance_history.clear()
        self.selection_history.clear()
        self.timing_history.clear()
        self.iteration_history.clear()

        # Initialize tracking for each optimizer if it's a hyperheuristic
        if hasattr(opt_model.optimizer, "optimizers"):
            for optimizer in opt_model.optimizer.optimizers:
                optimizer_name = optimizer.__class__.__name__
                self.performance_history[optimizer_name] = []
                self.selection_history[optimizer_name] = []
                self.timing_history[optimizer_name] = []

    def on_iteration_begin(self, iteration: int, opt_model: Opytimizer) -> None:
        """Track iteration start."""
        self.iteration_history.append(iteration)

        # Track current optimizer if it's a hyperheuristic
        if hasattr(opt_model.optimizer, "current_optimizer"):
            self.current_optimizer = opt_model.optimizer.current_optimizer

    def on_evaluate_after(self, *evaluate_args) -> None:
        """Track performance after evaluation."""
        if self.current_optimizer is None:
            return

        optimizer_name = self.current_optimizer.__class__.__name__
        space = evaluate_args[0]

        # Track performance (best fitness)
        performance = space.best_agent.fit

        if optimizer_name not in self.performance_history:
            self.performance_history[optimizer_name] = []

        self.performance_history[optimizer_name].append(performance)

        # Keep only the last window_size entries
        if len(self.performance_history[optimizer_name]) > self.window_size:
            self.performance_history[optimizer_name] = self.performance_history[
                optimizer_name
            ][-self.window_size :]

    def on_update_after(self, *update_args) -> None:
        """Track optimizer selection after update."""
        if self.current_optimizer is None:
            return

        optimizer_name = self.current_optimizer.__class__.__name__
        current_iteration = len(self.iteration_history) - 1

        # Track selection
        if optimizer_name not in self.selection_history:
            self.selection_history[optimizer_name] = []

        self.selection_history[optimizer_name].append(current_iteration)

    def get_best_performance(self, optimizer_name: str) -> Optional[float]:
        """Get the best performance achieved by an optimizer."""
        if optimizer_name not in self.performance_history:
            return None

        performances = self.performance_history[optimizer_name]
        if not performances:
            return None

        return min(performances)  # Assuming minimization problem

    def get_average_performance(self, optimizer_name: str) -> Optional[float]:
        """Get the average performance of an optimizer."""
        if optimizer_name not in self.performance_history:
            return None

        performances = self.performance_history[optimizer_name]
        if not performances:
            return None

        return np.mean(performances)

    def get_selection_frequency(self, optimizer_name: str) -> float:
        """Get the selection frequency of an optimizer."""
        if optimizer_name not in self.selection_history:
            return 0.0

        total_selections = len(self.selection_history[optimizer_name])
        total_iterations = len(self.iteration_history)

        if total_iterations == 0:
            return 0.0

        return total_selections / total_iterations

    def get_performance_ranking(self) -> List[tuple]:
        """Get performance ranking of all optimizers."""
        rankings = []

        for optimizer_name in self.performance_history:
            best_performance = self.get_best_performance(optimizer_name)
            if best_performance is not None:
                rankings.append((optimizer_name, best_performance))

        # Sort by performance (ascending for minimization)
        rankings.sort(key=lambda x: x[1])
        return rankings

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all optimizers."""
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
                "selection_frequency": self.get_selection_frequency(optimizer_name),
            }

        stats["optimizer_statistics"] = optimizer_stats
        return stats
