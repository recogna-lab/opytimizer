"""Base classes of the benchmark problems (not part of the public API)."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.environment import Environment

Bound = Union[float, int, Sequence[float]]


class _Benchmark(ABC):
    """Base class of every benchmark problem.

    A benchmark is a callable that can be handed directly to
    `opytimizer.core.Function`. It accepts a single solution, of shape
    (n_variables,) or (n_variables, 1), or a batch of solutions, of shape
    (N, n_variables) or (N, n_variables, n_dimensions), and computes on the
    array module (numpy / cupy) of its `env`.

    """

    def __init__(
        self,
        n_variables: int,
        n_objectives: int,
        lower_bound: Bound,
        upper_bound: Bound,
        env: Optional[Environment] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_objectives: Number of objectives.
            lower_bound: Lower bound of each variable (or a scalar shared by all).
            upper_bound: Upper bound of each variable (or a scalar shared by all).
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        if not isinstance(n_variables, (int, np.integer)) or n_variables <= 0:
            raise e.ValueError("`n_variables` should be a positive integer")
        if not isinstance(n_objectives, (int, np.integer)) or n_objectives <= 0:
            raise e.ValueError("`n_objectives` should be a positive integer")

        self.n_variables = int(n_variables)
        self.n_objectives = int(n_objectives)
        self.env = env if env is not None else Environment("numpy", "float64")

        lb = np.broadcast_to(np.asarray(lower_bound, dtype=float), (self.n_variables,))
        ub = np.broadcast_to(np.asarray(upper_bound, dtype=float), (self.n_variables,))
        if np.any(lb > ub):
            raise e.ValueError("`lower_bound` should be lower or equal to `upper_bound`")

        # Host copies feed `SearchSpace`; device copies are used by `_evaluate`
        self.lower_bound: List[float] = lb.tolist()
        self.upper_bound: List[float] = ub.tolist()
        self._lb = self.xp.asarray(lb, dtype=self.dtype)
        self._ub = self.xp.asarray(ub, dtype=self.dtype)

    @property
    def name(self) -> str:
        """Name of the problem."""
        return self.__class__.__name__

    @property
    def xp(self) -> Any:
        """Concrete array module (numpy / cupy) of the environment."""
        # Resolved lazily, so CuPy is only imported when a CuPy env is used
        return self.env.xp.__wrapped__

    @property
    def dtype(self) -> Any:
        """Floating dtype of the computations."""
        return self.xp.dtype(self.env.dtype)

    def __call__(self, x: Any) -> Any:
        """Evaluates the problem (see `evaluate`)."""
        return self.evaluate(x)

    def __repr__(self) -> str:
        return (
            f"{self.name}(n_variables={self.n_variables}, "
            f"n_objectives={self.n_objectives}, env={self.env!r})"
        )

    def evaluate(self, x: Any) -> Any:
        """Evaluates one or several solutions.

        Args:
            x: A solution of shape (n_variables,) or (n_variables, 1), or a batch of
                shape (N, n_variables) or (N, n_variables, n_dimensions).

        Returns:
            (Any): For a single solution, a scalar (single-objective) or an array of
                shape (n_objectives,); for a batch, an array of shape (N,) or
                (N, n_objectives). Arrays live on the environment's backend.

        """

        X = self._to_backend(x)
        X, single = self._as_batch(X)

        F = self._evaluate(X, self.xp)

        return F[0] if single else F

    def _to_backend(self, x: Any) -> Any:
        """Moves `x` to the environment's backend, cast to its dtype."""
        if not self.env.use_cuda and hasattr(x, "__cuda_array_interface__"):
            # CuPy forbids implicit device -> host conversions
            x = x.get()

        return self.xp.asarray(x, dtype=self.dtype)

    def _as_batch(self, X: Any):
        """Reshapes the input to (N, n_variables).

        Returns:
            (Tuple[Any, bool]): The batch and whether the input was a single solution.

        """

        n = self.n_variables

        if X.ndim == 1 and X.shape[0] == n:
            return X[None, :], True
        if X.ndim == 2 and X.shape[1] == n:
            return X, False
        # Position of a non-tensorized agent: (n_variables, 1)
        if X.ndim == 2 and X.shape == (n, 1):
            return X.reshape(1, n), True
        # Tensorized population: (N, n_variables, n_dimensions)
        if X.ndim == 3 and X.shape[1] * X.shape[2] == n:
            return X.reshape(X.shape[0], n), False

        raise e.SizeError(
            f"`x` with shape {tuple(X.shape)} is not compatible with "
            f"n_variables={n} of {self.name}"
        )

    def _normalize(self, X: Any) -> Any:
        """Maps the decision variables from [lower_bound, upper_bound] to [0, 1]."""
        return (X - self._lb) / (self._ub - self._lb)

    @abstractmethod
    def _evaluate(self, X: Any, xp: Any) -> Any:
        """Evaluates a batch of shape (N, n_variables).

        Args:
            X: Decision variables on the environment's backend.
            xp: Array module (numpy / cupy).

        Returns:
            (Any): Objective values of shape (N,) or (N, n_objectives).

        """


class _SingleObjectiveBenchmark(_Benchmark):
    """Base class of the single-objective problems.

    Subclasses define the standard search domain (`_default_bounds`), the optimal
    value and `_optimal_solution`.

    """

    # (lower, upper) bound shared by every variable in the literature
    _default_bounds = (-100.0, 100.0)

    def __init__(
        self,
        n_variables: int = 30,
        lower_bound: Optional[Bound] = None,
        upper_bound: Optional[Bound] = None,
        env: Optional[Environment] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            lower_bound: Lower bound(s). Defaults to the standard domain of the problem.
            upper_bound: Upper bound(s). Defaults to the standard domain of the problem.
            env: Environment that runs the problem. Defaults to numpy / float64.

        """

        lb = self._default_bounds[0] if lower_bound is None else lower_bound
        ub = self._default_bounds[1] if upper_bound is None else upper_bound

        super().__init__(n_variables, 1, lb, ub, env)

    @property
    def optimal_value(self) -> float:
        """Value of the global minimum."""
        return 0.0

    def optimal_solution(self) -> Any:
        """Returns a global minimizer, of shape (n_variables,), on the env's backend."""
        x = np.broadcast_to(
            np.asarray(self._optimal_solution(), dtype=float), (self.n_variables,)
        )

        return self.xp.asarray(x, dtype=self.dtype)

    def _optimal_solution(self) -> Union[float, np.ndarray]:
        """Global minimizer on the host (a scalar is shared by every variable)."""
        return 0.0


class _MultiObjectiveBenchmark(_Benchmark):
    """Base class of the multi- and many-objective problems."""

    def pareto_front(self, n_points: int = 1000) -> Any:
        """Returns a sample of the true Pareto front.

        Linear and spherical fronts are sampled on a Das-Dennis lattice. Fronts with
        dominated parts (disconnected, mixed, ...) are oversampled, filtered and
        reduced to well-spread points; with many objectives (M > 5) their position
        space is sampled sparsely, so a few of these points may be slightly dominated.

        Args:
            n_points: Maximum number of points. Lattice-based fronts return the
                largest lattice that fits, so they may return fewer points.

        Returns:
            (Any): Points of shape (P, n_objectives) on the environment's backend.

        """

        if not isinstance(n_points, (int, np.integer)) or n_points <= 0:
            raise e.ValueError("`n_points` should be a positive integer")

        front = self._pareto_front(int(n_points))

        return self.xp.asarray(front, dtype=self.dtype)

    @abstractmethod
    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Computes the front on the host, in float64, with shape (P, n_objectives)."""
