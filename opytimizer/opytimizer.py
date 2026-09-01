"""Optimization entry point.
"""

import time
from inspect import signature
from typing import Any, List, Optional, Union, Dict
from abc import abstractmethod

import dill

import opytimizer.utils.exception as e
from opytimizer.utils.exception import BudgetExhausted
from opytimizer.core.stopping import _StoppingVessel, MaxIterations, _StoppingCriterion
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer, MultiObjectiveOptimizer, TensorizedOptimizer, TensorizedMultiObjectiveOptimizer
from opytimizer.core.space import _SingleObjectiveSpace, _MultiObjectiveSpace, _SingleObjectiveTensorSpace, _MultiObjectiveTensorSpace
from opytimizer.utils import logging
from opytimizer.utils.callback import Callback, CallbackVessel
from opytimizer.utils.history import History
from opytimizer.math.metrics import BaseMetric

logger = logging.get_logger(__name__)

class _BaseRunner:
    """Shared pipeline: evaluate -> loop(update -> evaluate -> dump) -> finalize."""

    def __init__(
        self,
        space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace],
        optimizer: Optimizer,
        function: Function,
        save_agents: bool = False,
        save_history: bool = False
    ) -> None:
        
        logger.info("Creating class: %s.", self.__class__.__name__)
        self._ensure_space(space, optimizer)

        self.space = space
        self.optimizer = optimizer
        sync_method = getattr(optimizer, 'sync', None)
        self.sync_optimizer = sync_method if callable(sync_method) else (lambda *args, **kwargs: None)
        self.optimizer.compile(space=space)
        self.function = function
        self.save_history = save_history
        self.history = History(save_agents=save_agents)
        self.iteration = 0
        self.total_iterations = 0

        logger.debug(
            "Space: %s | Optimizer: %s | Function: %s.",
            self.space, self.optimizer, self.function,
        )
        logger.info("Class created.")

        
    @property
    def space(self) -> Union[_SingleObjectiveSpace, _MultiObjectiveSpace, _SingleObjectiveTensorSpace, _MultiObjectiveTensorSpace]:
        return self._space
 
    @space.setter
    def space(self, space):
        if not space.built:
            raise e.BuildError("`space` should be built before using Opytimizer")
        self._space = space
 
    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer
 
    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        if not optimizer.built:
            raise e.BuildError("`optimizer` should be built before using Opytimizer")
        self._optimizer = optimizer
 
    @property
    def function(self) -> Function:
        return self._function
 
    @function.setter
    def function(self, function: Function) -> None:
        if not function.built:
            raise e.BuildError("`function` should be built before using Opytimizer")
        self._function = function
 
    @property
    def history(self) -> History:
        return self._history
 
    @history.setter
    def history(self, history: History) -> None:
        if not isinstance(history, History):
            raise e.TypeError("`history` should be a History")
        self._history = history
 
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
 
    @property
    def total_iterations(self) -> int:
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
        args = signature(self.optimizer.evaluate).parameters
        return [getattr(self, v) for v in args]
 
    @property
    def update_args(self) -> List[Any]:
        args = signature(self.optimizer.update).parameters
        return [getattr(self, v) for v in args]


    @property
    def metrics(self) -> Optional[Union[List[BaseMetric], BaseMetric]]:
        return self._metrics

    @metrics.setter
    def metrics(self, m:Optional[Union[List[BaseMetric], BaseMetric]]) -> None:
        if isinstance(m, BaseMetric):
            self._metrics = [m]
        elif isinstance(m, List):
            if all(isinstance(m_i, BaseMetric) for m_i in m):
                self._metrics = m
        elif m is None:
            self._metrics = None
        else:
            raise e.TypeError('`metrics` should be an instance or a list of `BaseMetric` instances.')


    def _ensure_space(self, space: Union[_SingleObjectiveSpace, _SingleObjectiveTensorSpace,
                                        _MultiObjectiveSpace, _MultiObjectiveTensorSpace],
                                        optimizer: Union[Optimizer, MultiObjectiveOptimizer,
                                                         TensorizedOptimizer, TensorizedMultiObjectiveOptimizer]):
        """Ensure that the chosen space configuration is valid."""
        is_tensorized_optimizer = isinstance(optimizer, (TensorizedOptimizer, TensorizedMultiObjectiveOptimizer))
        is_tensorized_space = isinstance(space, (_SingleObjectiveTensorSpace, _MultiObjectiveTensorSpace))
        
        if is_tensorized_optimizer and not is_tensorized_space:
            msg = "The optimizer requires a tensorized space. Please set tensorized=True in the space configuration."
            logger.error(msg)
            raise TypeError(msg)

        if not is_tensorized_optimizer and is_tensorized_space:
            msg = "Standard optimizers cannot process a tensorized space. Use a TensorizedOptimizer or set tensorized=False."
            logger.error(msg)
            raise TypeError(msg)
         
    def evaluate(self, callbacks: CallbackVessel) -> None:
        callbacks.on_evaluate_before(*self.evaluate_args)
        self.optimizer.evaluate(*self.evaluate_args)
        callbacks.on_evaluate_after(*self.evaluate_args)
 
    def update(self, callbacks: CallbackVessel) -> None:
        callbacks.on_update_before(*self.update_args)
        self.optimizer.update(*self.update_args)
        callbacks.on_update_after(*self.update_args)
        self.space.clip_by_bound()

    def _compute_metrics(self) -> Dict:
        pass


    @abstractmethod
    def _dump_history(self, **kwargs) -> None:
        """Persist the iteration state to ``self.history``."""
 
    @abstractmethod
    def _set_postfix(self, stopping_vessel: _StoppingVessel, **kwargs) -> None:
        """Update the progress-bar postfix for this iteration."""
 
    @abstractmethod
    def _finalize_after_budget_exhaustion(self) -> None:
        """Ensure consistent state when the evaluation budget is exhausted."""


    def start(
        self,
        stopping_criteria=None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Union[List[BaseMetric], BaseMetric]] = None,
    ) -> None:
        """Start the optimization task."""
 
        logger.info("Starting optimization task.")
        self.n_iterations = getattr(stopping_criteria, "n", None)
 
        if stopping_criteria is None:
            stopping_criteria = [MaxIterations(250)]
        elif isinstance(stopping_criteria, _StoppingCriterion):
            stopping_criteria = [stopping_criteria]
 
        stopping_vessel = _StoppingVessel(stopping_criteria)
        callbacks = CallbackVessel(callbacks)
        self.metrics = metrics
        start = time.time()
        callbacks.on_task_begin(self)
        self.evaluate(callbacks)
        stopping_vessel.init_pbars()
        self.function.n_calls = 0
        try:
            while not stopping_vessel.should_stop(self):
                self.total_iterations += 1
                callbacks.on_iteration_begin(self.total_iterations, self)
 
                self.update(callbacks)
                self.evaluate(callbacks)
 
                stopping_vessel.update_pbars(self)
                metric_values = self._compute_metrics()
               
                self._set_postfix(stopping_vessel, metric_values=metric_values)
                self._dump_history(metric_values=metric_values)
 
                callbacks.on_iteration_end(self.total_iterations, self)
 
            stopping_vessel.update_pbars(self)
            
 
        except BudgetExhausted:
            logger.info(
                "Evaluation budget exhausted at %d calls. "
                "Finalizing with current population.",
                self.n_evals,
            )
            self._finalize_after_budget_exhaustion()
 
        finally:
            stopping_vessel.close_pbars()

        opt_time = time.time() - start
        self.sync_optimizer(self.space)        
        callbacks.on_task_end(self)
 
        self.history.dump(time=opt_time)
 
        logger.info("Optimization task ended.")
        logger.info("It took %s seconds.", opt_time)


    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            dill.dump(self, f)
 
    @classmethod
    def load(cls, file_path: str) -> "_BaseRunner":
        with open(file_path, "rb") as f:
            return dill.load(f)



class _SingleObjectiveRunner(_BaseRunner):
    """Pipeline for problems with a single objective function."""
 
    def _dump_history(self, **kwargs) -> None:
        if self.save_history:
            
            self.history.dump(
                agents=self.space.agents,
                best_agent=self.space.best_agent,
            )
            #logger.to_file(f"Fitness: {self.space.best_agent.fit}")
            #logger.to_file(f"Position: {self.space.best_agent.position}")
 
    def _set_postfix(self, stopping_vessel: _StoppingVessel, **kwargs) -> None:
        stopping_vessel.set_postfix(fitness=self.space.best_agent.fit)
 
    def _finalize_after_budget_exhaustion(self) -> None:
        # Nothing extra needed — history already has the latest state.
        pass


class _MultiObjectiveRunner(_BaseRunner):
    """Pipeline for problems with multiple objective functions."""
 
    def _compute_metrics(self) -> Dict:
        self.space.update_pareto_front(_xp=self.space.env.xp)
        if not self.metrics:
            return {}
 
        return {metric.name.lower(): metric(self.space.pareto_front) for metric in self.metrics}

    
    def _dump_history(self, metric_values: Dict, **kwargs) -> None:
    
        if self.save_history:
            self.history.dump(
                agents=self.space.agents,
                pareto_front=self.space.pareto_front,
                **metric_values,
            )
        logger.to_file(f"Pareto front size: {len(self.space.pareto_front)}")
            
        for name, value in metric_values.items():
            logger.to_file(f"{name}: {value}")
 
    def _set_postfix(self, stopping_vessel: _StoppingVessel, metric_values: Dict, **kwargs) -> None:
        if metric_values is not None:
            stopping_vessel.set_postfix(**metric_values)
 
    def _finalize_after_budget_exhaustion(self) -> None:
        self.space.update_pareto_front(_xp=self.space.env.xp)
        self.history.dump(
            agents=self.space.agents,
            pareto_front=self.space.pareto_front,
        )



class Opytimizer:

    def __new__(
        cls,
        space: Union[_SingleObjectiveSpace, _MultiObjectiveSpace],
        optimizer: Optimizer,
        function: Function,
        save_agents: bool = False,
        save_history: bool = False,
    ) -> _BaseRunner:
        runner_cls = (
            _SingleObjectiveRunner
            if space.n_objectives == 1
            else _MultiObjectiveRunner
        )
        return runner_cls(space, optimizer, function, save_agents, save_history)

    @classmethod
    def load(cls, file_path: str) -> _BaseRunner:
        with open(file_path, "rb") as f:
            return dill.load(f)
