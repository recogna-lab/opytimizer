"""Grid-based search space.
"""

import copy
from typing import Any, List, Optional, Tuple, Union

import opytimizer.utils.exception as e
from opytimizer.core import Environment
from opytimizer.core.space import (
    _MultiObjectiveSpace,
    _MultiObjectiveTensorSpace,
    _SingleObjectiveSpace,
    _SingleObjectiveTensorSpace,
)
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class _GridOps:
    def _build_grid(self, step) -> None:
        self.step = step
        self._create_grid()

    @property
    def step(self):
        """Step size of each variable."""

        return self._step

    @step.setter
    def step(self, step) -> None:
        step = self.env.xp.asarray(step, dtype=self.env.dtype)
        if not step.shape:
            step = self.env.xp.expand_dims(step, -1)
        if step.shape[0] != self.n_variables:
            raise e.SizeError("`step` should be the same size as `n_variables`")
        self._step = step

    @property
    def grid(self):
        """Grid with possible search values."""
        return self._grid

    @grid.setter
    def grid(self, grid) -> None:
        if not isinstance(grid, (float, int)) and not hasattr(grid, "shape"):
            raise e.TypeError("`grid` should be a valid value.")
        self._grid = grid

    def _create_grid(self) -> None:
        """Creates a grid of possible search values."""
        mesh = self.env.xp.meshgrid(
            *[
                self.step[i]
                * self.env.xp.arange(
                    float(self.lb[i] / self.step[i]),
                    float(self.ub[i] / self.step[i] + self.step[i]),
                )
                for i in range(self.n_variables)
            ]
        )
        self.grid = self.env.xp.stack([m.ravel() for m in mesh], dtype=self.env.dtype).T
        self.n_agents = len(self.grid)


class _SingleObjectiveGridSpace(_SingleObjectiveSpace, _GridOps):
    """A SingleObjective GridSpace class for agents, variables and methods
    related to the grid search space.

    """

    def __init__(
        self,
        n_variables: int,
        n_objectives: int,
        step: Union[float, List, Tuple, Any],
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_objectives: Number of objective functions.
            step: Variables' step.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        logger.info(
            "Overriding class: _SingleObjectiveSpace -> _SingleObjectiveGridSpace."
        )

        n_agents = 1
        n_dimensions = 1
        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )
        self._build_grid(step)
        self.build()

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""

        for agent, grid in zip(self.agents, self.grid):
            agent.fill_with_static(grid)

        self.best_agent = copy.deepcopy(self.agents[0])


class _MultiObjectiveGridSpace(_MultiObjectiveSpace, _GridOps):
    """A MultieObjective GridSpace class for agents, variables and methods
    related to the grid search space.

    """

    def __init__(
        self,
        n_variables: int,
        n_objectives: int,
        step: Union[float, List, Tuple, Any],
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_objectives: Number of objective functions.
            step: Variables' step.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.

        """

        logger.info(
            "Overriding class: _MultiObjectiveSpace -> _MultiObjectiveGridSpace."
        )

        n_agents = 1
        n_dimensions = 1
        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )
        self._build_grid(step)
        self.build()

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions."""

        for agent, grid in zip(self.agents, self.grid):
            agent.fill_with_static(grid)


class _SingleObjectiveTensorGridSpace(_SingleObjectiveTensorSpace, _GridOps):
    def __init__(
        self,
        n_variables: int,
        n_objectives: int,
        step: Union[float, List, Tuple, Any],
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _SingleObjectiveTensorSpace -> _SingleObjectiveTensorGridSpace."
        )

        n_agents = 1
        n_dimensions = 1
        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )
        self._build_grid(step)
        self.build()

    def _initialize_agents(self) -> None:
        self.X[:] = self.env.xp.asarray(self.grid, dtype=self.env.dtype).reshape(
            self.n_agents, self.n_variables, self.n_dimensions
        )


class _MultiObjectiveTensorGridSpace(_MultiObjectiveTensorSpace, _GridOps):
    def __init__(
        self,
        n_variables: int,
        n_objectives: int,
        step: Union[float, List, Tuple, Any],
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
    ) -> None:
        logger.info(
            "Overriding class: _MultiObjectiveTensorSpace -> _MultiObjectiveTensorGridSpace."
        )

        n_agents = 1
        n_dimensions = 1
        super().__init__(
            n_agents,
            n_variables,
            n_dimensions,
            n_objectives,
            lower_bound,
            upper_bound,
            mapping,
            env,
        )
        self._build_grid(step)
        self.build()

    def _initialize_agents(self) -> None:
        self.X[:] = self.env.xp.asarray(self.grid, dtype=self.env.dtype).reshape(
            self.n_agents, self.n_variables, self.n_dimensions
        )


class GridSpace:
    """
    A GridSpace Factory Class for agents, variables and methods
    related to the grid search space.
    """

    def __new__(
        self,
        n_variables: int,
        n_objectives: int,
        step: Union[float, List, Tuple, Any],
        lower_bound: Union[float, List, Tuple, Any],
        upper_bound: Union[float, List, Tuple, Any],
        mapping: Optional[List[str]] = None,
        env: Environment = None,
        tensorized: bool = False,
    ) -> Union[_SingleObjectiveGridSpace, _MultiObjectiveGridSpace]:
        """
        Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_objectives: Number of objective functions.
            step: Variables' step.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.
            env: Environment class object.
            tensorized: Boolean flag that indicates a tensorized population usage

        """
        if env is None:
            env = Environment("numpy", "float32")

        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be a positive value.")

        if tensorized:
            if n_objectives == 1:
                return _SingleObjectiveTensorGridSpace(
                    n_variables,
                    n_objectives,
                    step,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
            else:
                return _MultiObjectiveTensorGridSpace(
                    n_variables,
                    n_objectives,
                    step,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
        else:
            if n_objectives == 1:
                return _SingleObjectiveGridSpace(
                    n_variables,
                    n_objectives,
                    step,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
            else:
                return _MultiObjectiveGridSpace(
                    n_variables,
                    n_objectives,
                    step,
                    lower_bound,
                    upper_bound,
                    mapping,
                    env,
                )
