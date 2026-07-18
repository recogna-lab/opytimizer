"""Agent.
"""

import time
from typing import Dict, List, Optional, Union, Any

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.utils import logging
from opytimizer.core.environment import Environment

logger = logging.get_logger(__name__)


class Agent:
    """An Agent class for all optimization techniques."""

    def __init__(
        self,
        n_variables: int,
        n_dimensions: int,
        n_objectives: int,
        lower_bound: List[Union[int, float]],
        upper_bound: List[Union[int, float]],
        mapping: Optional[List[str]] = None,
        env: Environment = None
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions.
            n_objectives: Number of objective functions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.`
            env: Environment class object.

        """
        if env is None: env = Environment('cpu','float32')

        self.env = env
        self.xp = env.xp


        self.n_variables = n_variables
        self.n_dimensions = n_dimensions
        self.n_objectives = n_objectives

        self.lb = self.xp.asarray(lower_bound)
        self.ub = self.xp.asarray(upper_bound)

        self.position = self.xp.zeros((n_variables, n_dimensions), dtype=self.env.dtype)
        self._fit = self.xp.full((n_objectives,), self.xp.finfo(self.xp.float64).max, dtype=self.xp.float64).squeeze()

        self.mapping = mapping

        

        self.ts = int(time.time())

    @property
    def n_variables(self) -> int:
        """Number of decision variables."""
        return self._n_variables

    @n_variables.setter
    def n_variables(self, n_variables: int) -> None:
        if not isinstance(n_variables, int):
            raise e.TypeError("`n_variables` should be an integer")
        if n_variables <= 0:
            raise e.ValueError("`n_variables` should be > 0")

        self._n_variables = n_variables

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions."""
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions: int) -> None:
        if not isinstance(n_dimensions, int):
            raise e.TypeError("`n_dimensions` should be an integer")
        if n_dimensions <= 0:
            raise e.ValueError("`n_dimensions` should be > 0")

        self._n_dimensions = n_dimensions

    @property
    def position(self) -> np.ndarray:
        """N-dimensional array of positions."""

        return self._position

    @position.setter
    def position(self, position: Union[np.ndarray, Any]) -> None:

        self._position = position

    @property
    def n_objectives(self) -> int:
        """Number of objective functions."""
        return self._n_objectives

    @n_objectives.setter
    def n_objectives(self, n_objectives: int) -> None:
        if not isinstance(n_objectives, int):
            raise e.TypeError("`n_objectives` should be an integer")
        if n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be > 0")

        self._n_objectives = n_objectives

    @property
    def fit(self) -> Union[float, Any]:
        """Fitness value(s).

        Returns:
            Union[float, Any]: Single value for mono-objective or array for multi-objective.
        """
        return self._fit

    @fit.setter
    def fit(self, fit: Union[int, float, Any]) -> None:
        self._fit = self.xp.asarray(fit, dtype=self.xp.float64)

    @property
    def lb(self) -> np.ndarray:
        """Lower bounds."""
        return self._lb

    @lb.setter
    def lb(self, lb: Union[np.ndarray, Any]) -> None:
        if not lb.shape:
            lb = self.xp.expand_dims(lb, -1)
        if lb.shape[0] != self.n_variables:
            raise e.SizeError("`lb` should be the same size as `n_variables`")

        self._lb = lb

    @property
    def ub(self) -> np.ndarray:
        """Upper bounds."""
        return self._ub

    @ub.setter
    def ub(self, ub: Union[np.ndarray, Any]) -> None:
        
        if not ub.shape:
            ub = self.xp.expand_dims(ub, -1)
        if ub.shape[0] != self.n_variables:
            raise e.SizeError("`ub` should be the same size as `n_variables`")

        self._ub = ub

    @property
    def ts(self) -> int:
        """Timestamp of the agent."""

        return self._ts

    @ts.setter
    def ts(self, ts: int) -> None:
        if not isinstance(ts, int):
            raise e.TypeError("`ts` should be an integer")

        self._ts = ts

    @property
    def mapping(self) -> List[str]:
        """Variables mapping."""

        return self._mapping

    @mapping.setter
    def mapping(self, mapping: List[str]) -> None:
        if mapping is not None:
            if not isinstance(mapping, list):
                raise e.TypeError("`mapping` should be a list")
            if len(mapping) != self.n_variables:
                raise e.SizeError("`mapping` should be the same size as `n_variables`")
            self._mapping = mapping
        else:
            self._mapping = [f"x{i}" for i in range(self.n_variables)]
    
    @property
    def env(self) -> Environment:
        return self._env
    
    @env.setter
    def env(self, env_instance) -> None:
        if not isinstance(env_instance, Environment):
            raise e.TypeError("Error: please, pass a valiable environment.")
        self._env = env_instance

    @property
    def mapped_position(self) -> Dict[str, np.ndarray]:
        """Dictionary mapping variables names and array of positions."""

        return {m: p for (m, p) in zip(self.mapping, self.position)}

    def clip_by_bound(self) -> None:
        """Clips the agent's decision variables to the bounds limits."""

        self.position = self.xp.clip(
            self.position, 
            self.lb[:, self.xp.newaxis], 
            self.ub[:, self.xp.newaxis]
        )

    def fill_with_binary(self) -> None:
        """Fills the agent's decision variables with a binary distribution."""

        self.position = r.generate_binary_random_number(
            size=(self.n_variables, self.n_dimensions), 
            xp=self.xp
        )

    def fill_with_static(self, values: np.ndarray) -> None:
        """Fills the agent's decision variables with static values. Note that this
        method ignore the agent's bounds, so use it carefully.

        Args:
            values: Values to be filled (expected shape (n_variables,)).

        """

        values = self.xp.asarray(values)
        
        if values.ndim == 1:
            values = values[:, self.xp.newaxis]

        if values.shape[0] != self.n_variables:
            raise e.SizeError("`values` should be the same size as `n_variables`")
        
        self.position = self.xp.broadcast_to(values, self.position.shape).copy()

    def fill_with_uniform(self) -> None:
        """Fills the agent's decision variables with a uniform distribution
        based on bounds limits.

        """

        self.position = r.generate_uniform_random_number(
            low=self.lb[:, self.xp.newaxis], 
            high=self.ub[:, self.xp.newaxis], 
            size=(self.n_variables, self.n_dimensions), 
            xp=self.xp
        )

    def dominates(self, other: "Agent") -> bool:
        """Checks if this agent dominates another agent.

        Args:
            other: Another agent to be compared.

        Returns:
            (bool): Whether this agent dominates the other.

        """

        return self.xp.all(self._fit <= other._fit) and self.xp.any(self._fit < other._fit)