"""Block Evolutionary Algorithm (BlockEA / BEA)."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function, Optimizer, TensorizedOptimizer
from opytimizer.core.space import _SingleObjectiveSpace, _SingleObjectiveTensorSpace
from opytimizer.optimizers.single_objective.evolutionary._blocks import (
    _categorical,
    _regroup,
    _tournament,
)
from opytimizer.utils import logging

logger = logging.get_logger(__name__)

# Fixed graph of the report: 3 Tournament, 3 Exchange and 2 Crossover blocks
_N_TOURNAMENTS = 3
_N_EXCHANGES = 3
_N_CROSSOVERS = 2


def _bea_offspring(X: Any, F: Any, lb: Any, ub: Any, cfg: Dict, xp: Any) -> Any:
    """Creates the offspring of one BlockEA generation (Algorithm 1, lines 3-10).

    Args:
        X: Population of shape (N, D).
        F: Objective values of shape (N,).
        lb: Lower bounds of shape (D,).
        ub: Upper bounds of shape (D,).
        cfg: Block parameters prepared by `_BEABase._prepare`.
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Offspring of shape (L, D), clipped to the bounds.

    """

    D = X.shape[1]
    n_e, n_c = cfg["exchange_size"], cfg["crossover_size"]

    # Tournament blocks, whose parents are divided into three parts: Exchange
    # block j receives the j-th part of every tournament (P1j + P2j + P3j)
    P = X[_tournament(F, cfg["tournament_mask"], cfg["n_parents"], xp)]
    E = _regroup(P, _N_EXCHANGES)  # (3, T, D)

    # Exchange blocks (Algorithm 2): every n_e consecutive parents generate one
    # offspring, each variable copied from one parent sampled with (p_1..p_ne)
    groups = E.reshape(_N_EXCHANGES, -1, n_e, D)
    choice = _categorical(cfg["exchange_probabilities"], groups.shape[:2] + (D,), xp)
    E = xp.take_along_axis(groups, choice[:, :, None, :], axis=2)[:, :, 0, :]

    # Crossover 1 receives half of Exchange 1 and half of Exchange 2; Crossover 2
    # receives the whole output of Exchange 3
    half = E.shape[1] // 2
    C = xp.stack((xp.concatenate((E[0, :half], E[1, :half])), E[2]))

    # Crossover blocks (Algorithm 3): every n_c consecutive parents generate one
    # offspring o = sum_i w_i x_i, with (w_2..w_nc) = w1 * U[-1, 1] + w2 and
    # w_1 = 1 - (w_2 + ... + w_nc)
    groups = C.reshape(_N_CROSSOVERS, -1, n_c, D)
    R = xp.random.uniform(-1.0, 1.0, size=groups.shape[:2] + (n_c - 1, D))
    R = cfg["crossover_scales"] * R + cfg["crossover_shifts"]
    W = xp.concatenate((1.0 - R.sum(axis=2, keepdims=True), R), axis=2)
    O = xp.sum(W * groups, axis=2)

    return xp.clip(O.reshape(-1, D), lb, ub)


def _de_offspring(X: Any, F: Any, lb: Any, ub: Any, scale: float, rate: float, xp: Any) -> Any:
    """DE/current-to-best/1/bin trial vectors of the whole population.

    Args:
        X: Population of shape (N, D).
        F: Objective values of shape (N,).
        lb: Lower bounds of shape (D,).
        ub: Upper bounds of shape (D,).
        scale: Scaling factor F of the differences.
        rate: Crossover rate CR.
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Trial vectors of shape (N, D), clipped to the bounds.

    """

    N, D = X.shape
    best = X[xp.argmin(F)]
    r1 = xp.random.randint(0, N, size=N)
    r2 = xp.random.randint(0, N, size=N)

    V = X + scale * (best - X) + scale * (X[r1] - X[r2])

    # Binomial crossover, with at least one variable taken from the mutant
    take = xp.random.random((N, D)) < rate
    take[xp.arange(N), xp.random.randint(0, D, size=N)] = True

    return xp.clip(xp.where(take, V, X), lb, ub)


class _BEABase:
    """Hyperparameters, parameters and graph bookkeeping shared by both variants."""

    def _set_blocks(
        self,
        exchange_size: int,
        crossover_size: int,
        tournament_sizes: Optional[Sequence[int]],
        exchange_probabilities: Optional[Any],
        crossover_scales: Optional[Any],
        crossover_shifts: Optional[Any],
        n_offspring: Optional[int],
        max_generations: Optional[int],
        de_start: float,
        de_scale: float,
        de_rate: float,
    ) -> None:
        """Validates and stores the block parameters (see the optimizer docstrings)."""

        if not isinstance(exchange_size, (int, np.integer)) or exchange_size < 1:
            raise e.ValueError("`exchange_size` should be an integer >= 1")
        if not isinstance(crossover_size, (int, np.integer)) or crossover_size < 2:
            raise e.ValueError("`crossover_size` should be an integer >= 2")
        if max_generations is not None and (not isinstance(max_generations, (int, np.integer)) or max_generations <= 0):
            raise e.ValueError("`max_generations` should be a positive integer")
        if not 0.0 <= de_start <= 1.0:
            raise e.ValueError("`de_start` should be in [0, 1]")
        if de_scale <= 0 or not 0.0 <= de_rate <= 1.0:
            raise e.ValueError("`de_scale` should be positive and `de_rate` in [0, 1]")

        self.exchange_size = int(exchange_size)
        self.crossover_size = int(crossover_size)
        self.n_offspring = n_offspring
        self.max_generations = max_generations
        self.de_start = float(de_start)
        self.de_scale = float(de_scale)
        self.de_rate = float(de_rate)

        m = np.full(_N_TOURNAMENTS, 2) if tournament_sizes is None else np.asarray(tournament_sizes)
        if m.shape != (_N_TOURNAMENTS,) or np.any(m < 1) or np.any(m > 10):
            raise e.ValueError("`tournament_sizes` should have one value in [1, 10] per tournament block")
        self.tournament_sizes = m.astype(int)

        p = np.ones(self.exchange_size) if exchange_probabilities is None else exchange_probabilities
        p = np.broadcast_to(np.asarray(p, dtype=float), (_N_EXCHANGES, self.exchange_size))
        if np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
            raise e.ValueError("`exchange_probabilities` should be non-negative with a positive sum")
        self.exchange_probabilities = p / p.sum(axis=1, keepdims=True)

        # w1 (spread) and w2 (center) of the crossover weights of each block
        scales = np.array([0.5, 0.25]) if crossover_scales is None else crossover_scales
        shifts = np.zeros(_N_CROSSOVERS) if crossover_shifts is None else crossover_shifts
        self.crossover_scales = np.broadcast_to(np.asarray(scales, dtype=float), (_N_CROSSOVERS,)).copy()
        self.crossover_shifts = np.broadcast_to(np.asarray(shifts, dtype=float), (_N_CROSSOVERS,)).copy()

        self._generation = 0

    def _graph_sizes(self, n_agents: int):
        """Parents per tournament block, so that every division is exact.

        Returns:
            (Tuple[int, int]): Parents per tournament block and offspring per generation.

        """

        target = n_agents if self.n_offspring is None else int(self.n_offspring)
        n_e, n_c = self.exchange_size, self.crossover_size

        def valid(t):
            # 3 equal parts, groups of n_e, halves of the exchange output and
            # groups of n_c in both crossover blocks
            return t % _N_EXCHANGES == 0 and t % n_e == 0 and (t // n_e) % (2 * n_c) == 0

        t = _N_EXCHANGES
        while not (valid(t) and 2 * t // (n_e * n_c) >= target):
            t += _N_EXCHANGES

        return t, 2 * t // (n_e * n_c)

    def _prepare(self, n_agents: int, xp: Any, dtype: Any) -> Dict:
        """Builds the block parameters on the backend."""
        n_parents, n_offspring = self._graph_sizes(n_agents)
        self.n_offspring_per_generation = n_offspring

        m_max = int(self.tournament_sizes.max())
        mask = np.arange(m_max)[None, None, :] < self.tournament_sizes[:, None, None]

        return {
            "n_parents": n_parents,
            "tournament_mask": xp.asarray(mask),
            "exchange_size": self.exchange_size,
            "exchange_probabilities": xp.asarray(self.exchange_probabilities, dtype=dtype),
            "crossover_size": self.crossover_size,
            "crossover_scales": xp.asarray(self.crossover_scales, dtype=dtype).reshape(-1, 1, 1, 1),
            "crossover_shifts": xp.asarray(self.crossover_shifts, dtype=dtype).reshape(-1, 1, 1, 1),
        }

    def _de_active(self) -> bool:
        """Whether the late-stage DE is connected in the current generation."""
        if self.max_generations is None:
            return False

        return self._generation >= self.de_start * self.max_generations


_INIT_DOC = """
        Args:
            params: Contains key-value parameters to the meta-heuristics.
            exchange_size: Parents per Exchange offspring (nParents of Algorithm 2).
            crossover_size: Parents per Crossover offspring (nParents of Algorithm 3).
            tournament_sizes: m of each of the 3 Tournament blocks, in [1, 10].
            exchange_probabilities: Probability of each parent in the Exchange
                blocks, shape (3, exchange_size) (broadcastable).
            crossover_scales: w1 of each of the 2 Crossover blocks.
            crossover_shifts: w2 of each of the 2 Crossover blocks.
            n_offspring: Minimum number of offspring per generation (defaults to
                the population size); it is rounded up so that every division
                of the graph is exact.
            max_generations: Number of generations of the run. When given, the
                late-stage DE is connected after `de_start * max_generations`.
            de_start: Fraction of the run after which the DE is connected.
            de_scale: Scaling factor F of the late-stage DE.
            de_rate: Crossover rate CR of the late-stage DE.
"""


class BEA(_BEABase, Optimizer):
    """Block Evolutionary Algorithm (BlockEA), single-objective.

    Three Tournament blocks select parents, which are divided among three
    Exchange blocks (each variable copied from one of `exchange_size` parents);
    half of the output of Exchange 1 and 2 feeds Crossover 1 and the output of
    Exchange 3 feeds Crossover 2, both combining `crossover_size` parents with
    weights that sum to one. The environmental selection keeps the N best of
    the population and the offspring. In the late stage, a DE/current-to-best/
    1/bin is connected to speed up the convergence.

    The report does not list the trained block parameters nor the details of the
    late-stage DE, so they are exposed as arguments.

    References:
        X. Qi, Z. Yang, S. Zhu, H. Geng, S. Yang, and Y. Tian, "Report for Block
        Evolutionary Algorithm," competition report.

    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        exchange_size: int = 2,
        crossover_size: int = 3,
        tournament_sizes: Optional[Sequence[int]] = None,
        exchange_probabilities: Optional[Any] = None,
        crossover_scales: Optional[Any] = None,
        crossover_shifts: Optional[Any] = None,
        n_offspring: Optional[int] = None,
        max_generations: Optional[int] = None,
        de_start: float = 0.7,
        de_scale: float = 0.5,
        de_rate: float = 0.9,
    ) -> None:
        logger.info("Overriding class: Optimizer -> BEA (Default).")

        super().__init__()

        self._set_blocks(
            exchange_size, crossover_size, tournament_sizes, exchange_probabilities,
            crossover_scales, crossover_shifts, n_offspring, max_generations,
            de_start, de_scale, de_rate,
        )  # fmt: skip

        self.build(params)
        self._cfg = None

        logger.info("Class overrided.")

    __init__.__doc__ = "Initialization method.\n" + _INIT_DOC

    def compile(self, space: _SingleObjectiveSpace) -> None:
        """Prepares the block parameters for the space."""
        self._cfg = self._prepare(space.n_agents, np, float)
        self._lb = np.asarray(space.agents[0].lb, dtype=float).reshape(-1)
        self._ub = np.asarray(space.agents[0].ub, dtype=float).reshape(-1)
        self._generation = 0

    def evaluate(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """Evaluates the initial population (offspring are evaluated in `update`)."""
        for agent in space.agents:
            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

        self.evaluate = lambda: None

    def update(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """Performs one generation: offspring creation and environmental selection.

        Args:
            space: A SingleObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        agents = space.agents
        shape = agents[0].position.shape
        X = np.stack([a.position.reshape(-1) for a in agents])
        F = np.array([float(np.asarray(a.fit).reshape(-1)[0]) for a in agents])

        O = _bea_offspring(X, F, self._lb, self._ub, self._cfg, np)
        if self._de_active():
            de = _de_offspring(X, F, self._lb, self._ub, self.de_scale, self.de_rate, np)
            O = np.concatenate((O, de))

        F_O = np.array([float(np.asarray(function(o.reshape(shape))).reshape(-1)[0]) for o in O])

        # Environmental selection: the N best of the population and the offspring
        F_all = np.concatenate((F, F_O))
        best = np.argsort(F_all, kind="stable")[: len(agents)]

        X_all = np.concatenate((X, O))
        for agent, i in zip(agents, best):
            agent.position = X_all[i].reshape(shape).copy()
            agent.fit = F_all[i]

        if agents[0].fit < space.best_agent.fit:
            space.best_agent.position = copy.deepcopy(agents[0].position)
            space.best_agent.fit = copy.deepcopy(agents[0].fit)
            space.best_agent.ts = int(time.time())

        self._generation += 1


class BEATensor(_BEABase, Optimizer, TensorizedOptimizer):
    """Tensorized (NumPy / CuPy) BlockEA: the same graph, run on the whole population."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        exchange_size: int = 2,
        crossover_size: int = 3,
        tournament_sizes: Optional[Sequence[int]] = None,
        exchange_probabilities: Optional[Any] = None,
        crossover_scales: Optional[Any] = None,
        crossover_shifts: Optional[Any] = None,
        n_offspring: Optional[int] = None,
        max_generations: Optional[int] = None,
        de_start: float = 0.7,
        de_scale: float = 0.5,
        de_rate: float = 0.9,
    ) -> None:
        logger.info("Overriding class: Optimizer -> BEA (Tensor).")

        super().__init__()

        self._set_blocks(
            exchange_size, crossover_size, tournament_sizes, exchange_probabilities,
            crossover_scales, crossover_shifts, n_offspring, max_generations,
            de_start, de_scale, de_rate,
        )  # fmt: skip

        self.build(params)
        self._cfg = None
        self._xp = None

        logger.info("Class overrided.")

    __init__.__doc__ = "Initialization method.\n" + _INIT_DOC

    def compile(self, space: _SingleObjectiveTensorSpace) -> None:
        """Prepares the block parameters on the space's backend."""
        xp = space.env.xp.__wrapped__
        self._xp = xp
        self.dtype = xp.dtype(space.env.dtype)

        self._cfg = self._prepare(space.n_agents, xp, self.dtype)
        self._lb = xp.asarray(space.lb, dtype=self.dtype).reshape(-1)
        self._ub = xp.asarray(space.ub, dtype=self.dtype).reshape(-1)
        self._generation = 0

    def _update_best(self, space: _SingleObjectiveTensorSpace) -> None:
        """Copies the best row into `best_agent`.

        The selection is elitist and sorted, so row 0 is always the best solution
        found so far (`global_best_*` are read by `TensorizedOptimizer.sync`).
        """

        self.global_best_position = space.X[0].copy()
        self.global_best_fit = space.F[0].copy()

        space.best_agent.position = self.global_best_position
        space.best_agent.fit = self.global_best_fit
        space.best_agent.ts = int(time.time())

    def evaluate(self, space: _SingleObjectiveTensorSpace, function: Function) -> None:
        """Evaluates the initial population (offspring are evaluated in `update`)."""
        xp = self._xp

        space.F = function(space.X, xp=xp).reshape(-1)
        order = xp.argsort(space.F)
        space.X, space.F = space.X[order], space.F[order]
        self._update_best(space)

        self.evaluate = lambda: None

    def update(self, space: _SingleObjectiveTensorSpace, function: Function) -> None:
        """Performs one generation: offspring creation and environmental selection.

        Args:
            space: A tensorized SingleObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        xp = self._xp
        N = space.X.shape[0]
        feat_shape = space.X.shape[1:]

        X = space.X.reshape(N, -1)
        O = _bea_offspring(X, space.F, self._lb, self._ub, self._cfg, xp)
        if self._de_active():
            de = _de_offspring(X, space.F, self._lb, self._ub, self.de_scale, self.de_rate, xp)
            O = xp.concatenate((O, de))
        O = O.astype(self.dtype)

        F_O = function(O.reshape((O.shape[0],) + feat_shape), xp=xp).reshape(-1).astype(self.dtype)

        # Environmental selection: the N best of the population and the offspring
        F_all = xp.concatenate((space.F, F_O))
        best = xp.argsort(F_all)[:N]

        space.X = xp.concatenate((X, O))[best].reshape((N,) + feat_shape)
        space.F = F_all[best]
        self._update_best(space)

        self._generation += 1
