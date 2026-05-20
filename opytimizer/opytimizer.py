"""Optimization entry point.
"""

import time
from inspect import signature
from typing import Any, List, Optional, Union

import dill
import numpy as np

import opytimizer.utils.exception as e
from opytimizer.utils.exception import BudgetExhausted
from opytimizer.core.stopping import _StoppingVessel, MaxIterations, _StoppingCriterion
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import _SingleObjectiveSpace, _MultiObjectiveSpace
from opytimizer.utils import logging
from opytimizer.utils.callback import Callback, CallbackVessel
from opytimizer.utils.history import History

logger = logging.get_logger(__name__)


class Opytimizer:
    """An Opytimizer class holds all the information needed
    in order to perform an optimization task.

    """

    def __init__(
        self,
        space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace],
        optimizer: Optimizer,
        function: Function,
        save_agents: bool = False,
    ) -> None:
        """Initialization method.

        Args:
            space: Space-child instance.
            optimizer: Optimizer-child instance.
            function: Function or Function-child instance.
            save_agents: Saves all agents in the search space.

        """

        logger.info("Creating class: Opytimizer.")

        self.space = space

        self.optimizer = optimizer
        self.optimizer.compile(space)

        self.function = function

        self.history = History(save_agents=save_agents)

        self.iteration = 0
        self.total_iterations = 0

        logger.debug(
            "Space: %s | Optimizer: %s| Function: %s.",
            self.space,
            self.optimizer,
            self.function,
        )
        logger.info("Class created.")

    @property
    def space(self) -> Union[_SingleObjectiveSpace, _MultiObjectiveSpace]:
        """Space-child instance (SearchSpace, HyperComplexSpace, etc)."""

        return self._space

    @space.setter
    def space(self, space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace]) -> None:
        if not space.built:
            raise e.BuildError("`space` should be built before using Opytimizer")

        self._space = space

    @property
    def optimizer(self) -> Optimizer:
        """Optimizer-child instance (PSO, BA, etc)."""

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        if not optimizer.built:
            raise e.BuildError("`optimizer` should be built before using Opytimizer")

        self._optimizer = optimizer

    @property
    def function(self) -> Function:
        """Function or Function-child instance (ConstrainedFunction, WeightedFunction, etc)."""

        return self._function

    @function.setter
    def function(self, function: Function) -> None:
        if not function.built:
            raise e.BuildError("`function` should be built before using Opytimizer")

        self._function = function

    @property
    def history(self) -> History:
        """Optimization history."""

        return self._history

    @history.setter
    def history(self, history: History) -> None:
        if not isinstance(history, History):
            raise e.TypeError("`history` should be a History")

        self._history = history

    @property
    def iteration(self) -> int:
        """Current iteration."""

        return self._iteration

    @iteration.setter
    def iteration(self, iteration: int) -> None:
        if not isinstance(iteration, int):
            raise e.TypeError("`iteration` should be an integer")
        if iteration < 0:
            raise e.ValueError("`iteration` should be >= 0")

        self._iteration = iteration

    @property
    def total_iterations(self) -> int:
        """Total number of iterations."""

        return self._total_iterations

    @total_iterations.setter
    def total_iterations(self, total_iterations: int) -> None:
        if not isinstance(total_iterations, int):
            raise e.TypeError("`total_iterations` should be an integer")
        if total_iterations < 0:
            raise e.ValueError("`total_iterations` should be >= 0")

        self._total_iterations = total_iterations

    @property
    def n_evals(self) -> int:
        return self.function.n_calls

    @property
    def evaluate_args(self) -> List[Any]:
        """Converts the optimizer `evaluate` arguments into real variables.

        Returns:
            (List[Any]): List of real-attribute variables.

        """

        args = signature(self.optimizer.evaluate).parameters

        return [getattr(self, v) for v in args]

    @property
    def update_args(self) -> List[Any]:
        """Converts the optimizer `update` arguments into real variables.

        Returns:
            (List[Any]): List of real-attribute variables.

        """

        args = signature(self.optimizer.update).parameters

        return [getattr(self, v) for v in args]

    def evaluate(self, callbacks: List[Callback]) -> None:
        """Wraps the `evaluate` pipeline with its corresponding callbacks.

        Args:
            callbacks: List of callbacks.

        """

        callbacks.on_evaluate_before(*self.evaluate_args)
        self.optimizer.evaluate(*self.evaluate_args)
        callbacks.on_evaluate_after(*self.evaluate_args)

    def update(self, callbacks: List[Callback]) -> None:
        """Wraps the `update` pipeline with its corresponding callbacks.

        Args:
            callback: List of callbacks.

        """

        callbacks.on_update_before(*self.update_args)
        self.optimizer.update(*self.update_args)
        callbacks.on_update_after(*self.update_args)

        # Regardless of callbacks or not, every update on the search space
        # must meet the bounds limits
        self.space.clip_by_bound()

    def start(
        self,
        stopping_criteria=None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[List] = None,
    ) -> None:
        """Starts the optimization task.

        Args:
            n_iterations: Maximum number of iterations.
            stopping_criteria: List of stopping criteria
            callbacks: List of callbacks.
            metrics: List of metric callables (opcional, apenas para MOO).
        """

        logger.info("Starting optimization task.")
        self.n_iterations = getattr(stopping_criteria, "n", None)
        
        if stopping_criteria is None:
            stopping_criteria = [MaxIterations(250)]

        elif isinstance(stopping_criteria, _StoppingCriterion):
            stopping_criteria = [stopping_criteria]

        stopping_vessel = _StoppingVessel(stopping_criteria)
        callbacks = CallbackVessel(callbacks)

        start = time.time()

        callbacks.on_task_begin(self)

        self.evaluate(callbacks)

        stopping_vessel.init_pbars()
        
        try:
            while not stopping_vessel.should_stop(self):
           
                self.total_iterations += 1
                
                callbacks.on_iteration_begin(self.total_iterations, self)

                self.update(callbacks)
                self.evaluate(callbacks)

                stopping_vessel.update_pbars(self)

                if self.space.n_objectives == 1:
                    stopping_vessel.set_postfix(fitness=self.space.best_agent.fit)
                    self.history.dump(
                        agents=self.space.agents, best_agent=self.space.best_agent
                        )   
                    logger.to_file(f"Fitness: {self.space.best_agent.fit}")
                    logger.to_file(f"Position: {self.space.best_agent.position}")
                else:
                    pareto_front = [ag.fit for ag in self.space.pareto_front]
                    pareto_front = self.space.env.xp.asarray(pareto_front)
                    metric_values = {}
                    
                    if metrics:
                        for metric in metrics:
                            value = metric(pareto_front)
                            metric_values[metric.name.lower()] = value
                        stopping_vessel.set_postfix(**metric_values)
                    
                    self.history.dump(
                        agents=self.space.agents,
                        pareto_front=self.space.pareto_front,
                        metric_values=metric_values,
                        )
                    logger.to_file(f"Pareto front size: {len(self.space.pareto_front)}")
                    for name, value in metric_values.items():
                        logger.to_file(f"{name}: {value}")
            stopping_vessel.update_pbars(self)
            callbacks.on_iteration_end(self.total_iterations, self)
            
        except BudgetExhausted:
            logger.info("Evaluation budget exhausted at %d calls. "
                            "Finalizing with current population.",
                            self.n_evals
                            )
                
            self._finalize_after_budget_exhaustion()

        finally:
            stopping_vessel.close_pbars()

        callbacks.on_task_end(self)

        end = time.time()
        
        if hasattr(self.space, 'update_pareto_front'): self.space.update_pareto_front()    
        
        opt_time = end - start

        self.history.dump(time=opt_time)

        logger.info("Optimization task ended.")
        logger.info("It took %s seconds.", opt_time)


    def _finalize_after_budget_exhaustion(self) -> None:
        """Ensures consistent state when budget is hit mid-iteration."""
        if self.space.n_objectives > 1:
            self.space.update_pareto_front(self.space.agents)
            self.history.dump(
                agents=self.space.agents,
                pareto_front=self.space.pareto_front)

        

    def save(self, file_path: str) -> None:
        """Saves the optimization model to a dill (pickle) file.

        Args:
            file_path: Path of file to be saved.

        """

        with open(file_path, "wb") as output_file:
            dill.dump(self, output_file)

    @classmethod
    def load(cls, file_path: str) -> None:
        """Loads the optimization model from a dill (pickle) file without needing
        to instantiate the class.

        Args:
            file_path: Path of file to be loaded.

        """

        with open(file_path, "rb") as input_file:
            opt_model = dill.load(input_file)

            return opt_model
