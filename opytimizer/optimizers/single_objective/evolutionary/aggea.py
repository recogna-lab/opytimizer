"""Automatic Generated Graph Evolutionary Algorithm (AG-GEA)."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function, Optimizer, TensorizedOptimizer
from opytimizer.core.space import _SingleObjectiveSpace, _SingleObjectiveTensorSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


def _categorical(probabilities: Any, shape: tuple, xp: Any) -> Any:
    """Samples categories from per-block probabilities.

    Args:
        probabilities: Normalized probabilities of shape (B, K).
        shape: Sample shape (B, ...), whose leading axis indexes the blocks.
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Integer categories in [0, K) of shape `shape`.

    """

    cdf = xp.cumsum(probabilities, axis=1)
    cdf = cdf.reshape((cdf.shape[0],) + (1,) * (len(shape) - 1) + (cdf.shape[1],))
    u = xp.random.random(shape + (1,))

    return xp.minimum((u > cdf).sum(axis=-1), probabilities.shape[1] - 1)


def _partners(X: Any, n_partners: int, xp: Any) -> Any:
    """Each row of every block followed by `n_partners` random rows of the same block.

    Args:
        X: Solutions of shape (B, S, D), one group of S solutions per block.
        n_partners: Number of extra parents per offspring.
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Parents of shape (B, S, n_partners + 1, D); index 0 is the row itself.

    """

    B, S, _ = X.shape
    idx = xp.random.randint(0, S, size=(B, S, n_partners))
    others = X[xp.arange(B)[:, None, None], idx]

    return xp.concatenate((X[:, :, None, :], others), axis=2)


def _regroup(X: Any, n_blocks: int) -> Any:
    """Equally divides every block among `n_blocks` next blocks.

    Block j of the output receives the j-th chunk of every input block, so each
    next block mixes solutions coming from all previous blocks.

    Args:
        X: Solutions of shape (B, S, D), with S divisible by `n_blocks`.
        n_blocks: Number of next blocks.

    Returns:
        (Any): Solutions of shape (n_blocks, B * S / n_blocks, D).

    """

    B, S, D = X.shape
    chunk = S // n_blocks
    X = X.reshape(B, n_blocks, chunk, D).transpose(1, 0, 2, 3)

    return X.reshape(n_blocks, B * chunk, D)


def _aggea_offspring(X: Any, F: Any, lb: Any, ub: Any, cfg: Dict, xp: Any) -> Any:
    """Creates the offspring of one AG-GEA generation (Algorithm 1, lines 3-11).

    Args:
        X: Population of shape (N, D).
        F: Objective values of shape (N,).
        lb: Lower bounds of shape (D,).
        ub: Upper bounds of shape (D,).
        cfg: Block parameters prepared by `_AGGEABase._prepare`.
        xp: Array module (numpy / cupy).

    Returns:
        (Any): Offspring of shape (L, D), clipped to the bounds.

    """

    N, D = X.shape

    # Tournament blocks: each one selects `n_parents` parents from the whole
    # (duplicated) population by m-tournament, m differing among blocks
    B_t, n_parents, m_max = cfg["n_tournaments"], cfg["n_parents"], cfg["m_max"]
    candidates = xp.random.randint(0, N, size=(B_t, n_parents, m_max))
    fit = xp.where(cfg["tournament_mask"], F[candidates], xp.inf)
    winners = xp.take_along_axis(candidates, fit.argmin(axis=2)[..., None], axis=2)
    P = X[winners[..., 0]]  # (B_t, n_parents, D)

    # Exchange blocks: each variable is copied from one of n_e parents
    E = _regroup(P, cfg["n_exchanges"])
    parents = _partners(E, cfg["exchange_size"] - 1, xp)
    choice = _categorical(cfg["exchange_probabilities"], E.shape, xp)
    E = xp.take_along_axis(parents, choice[:, :, None, :], axis=2)[:, :, 0, :]

    # Crossover blocks: o = sum_i w_i x_i with w_1 = 1 - (w_2 + ... + w_nc), i.e.,
    # o = x_1 + sum_{i>1} w_i (x_i - x_1), each w_i drawn from a Gaussian mixture
    C = _regroup(E, cfg["n_crossovers"])
    parents = _partners(C, cfg["crossover_size"] - 1, xp)
    B_c, S_c = C.shape[:2]
    component = _categorical(cfg["crossover_weights"], (B_c, S_c, D), xp)
    blocks = xp.arange(B_c)[:, None, None]
    mu = cfg["crossover_means"][blocks, component]  # (B_c, S_c, D, nc - 1)
    sigma = cfg["crossover_stds"][blocks, component]
    w = mu + sigma * xp.random.standard_normal(mu.shape)
    diff = parents[:, :, 1:, :] - parents[:, :, :1, :]  # (B_c, S_c, nc - 1, D)
    C = C + xp.sum(w.transpose(0, 1, 3, 2) * diff, axis=2)

    # Mutation blocks: the rows of each crossover block are split among them by
    # ratio; o_k = x_k + w_0 (u_k - l_k), w_0 drawn from a zero-mean mixture
    target = cfg["mutation_routing"]  # (B_c, S_c) mutation block of each row
    stds = cfg["mutation_stds"][target]  # (B_c, S_c, s_m)
    weights = cfg["mutation_weights"][target]
    flat_weights = weights.reshape(-1, weights.shape[-1])
    component = _categorical(flat_weights, (B_c * S_c, D), xp).reshape(B_c, S_c, D)
    sigma = xp.take_along_axis(stds, component, axis=2)
    C = C + sigma * xp.random.standard_normal(C.shape) * (ub - lb)

    return xp.clip(C.reshape(-1, D), lb, ub)


class _AGGEABase:
    """Hyperparameters, parameters and graph bookkeeping shared by both variants."""

    def _set_blocks(
        self,
        n_tournaments: int,
        n_exchanges: int,
        n_crossovers: int,
        n_mutations: int,
        exchange_size: int,
        crossover_size: int,
        tournament_sizes: Optional[Sequence[int]],
        exchange_probabilities: Optional[Any],
        crossover_means: Optional[Any],
        crossover_stds: Optional[Any],
        crossover_weights: Optional[Any],
        mutation_stds: Optional[Any],
        mutation_weights: Optional[Any],
        crossover_split: Optional[Any],
        n_offspring: Optional[int],
    ) -> None:
        """Validates and stores the graph (see the optimizer docstrings)."""

        for name, value, minimum in (
            ("n_tournaments", n_tournaments, 1),
            ("n_exchanges", n_exchanges, 1),
            ("n_crossovers", n_crossovers, 1),
            ("n_mutations", n_mutations, 1),
            ("exchange_size", exchange_size, 1),
            ("crossover_size", crossover_size, 2),
        ):
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise e.ValueError(f"`{name}` should be an integer >= {minimum}")

        self.n_tournaments = int(n_tournaments)
        self.n_exchanges = int(n_exchanges)
        self.n_crossovers = int(n_crossovers)
        self.n_mutations = int(n_mutations)
        self.exchange_size = int(exchange_size)
        self.crossover_size = int(crossover_size)
        self.n_offspring = n_offspring

        # Tournament: m in [1, 10] per block (binary tournaments by default)
        m = np.full(self.n_tournaments, 2) if tournament_sizes is None else np.asarray(tournament_sizes)
        if m.shape != (self.n_tournaments,) or np.any(m < 1) or np.any(m > 10):
            raise e.ValueError("`tournament_sizes` should have one value in [1, 10] per tournament block")
        self.tournament_sizes = m.astype(int)

        # Exchange: probabilities (p_1, ..., p_ne) per block (uniform by default)
        p = (
            np.full((self.n_exchanges, self.exchange_size), 1.0)
            if exchange_probabilities is None
            else np.asarray(exchange_probabilities, dtype=float)
        )
        p = np.broadcast_to(p, (self.n_exchanges, self.exchange_size))
        self.exchange_probabilities = self._normalize(p, "exchange_probabilities")

        # Crossover: Gaussian mixture of the weights (w_2, ..., w_nc) per block.
        # Defaults: DE/rand/1 weights (w_2 ~ N(0.5, 0.1), w_3 ~ N(-0.5, 0.1)) or
        # a copy of the main parent, which reproduces a binomial crossover
        n_w = self.crossover_size - 1
        if crossover_means is None:
            de = np.zeros(n_w)
            de[0], de[-1] = 0.5, -0.5 if n_w > 1 else 0.5
            crossover_means = np.stack([de, np.zeros(n_w)])
        if crossover_stds is None:
            crossover_stds = np.stack([np.full(n_w, 0.1), np.zeros(n_w)])
        means = np.asarray(crossover_means, dtype=float)
        stds_arr = np.asarray(crossover_stds, dtype=float)
        # Number of mixture components: the component axis of any 2-D+ argument
        n_comp = max(
            [a.shape[-2] for a in (means, stds_arr) if a.ndim >= 2]
            + ([np.asarray(crossover_weights).shape[-1]] if crossover_weights is not None else [])
            + [1]
        )
        if crossover_weights is None:
            crossover_weights = [[0.9, 0.1], [0.5, 0.5]][: self.n_crossovers] if n_comp == 2 else np.ones(n_comp)
        shape = (self.n_crossovers, n_comp, n_w)
        try:
            self.crossover_means = np.broadcast_to(means, shape).copy()
            self.crossover_stds = np.broadcast_to(stds_arr, shape).copy()
            weights = np.broadcast_to(np.asarray(crossover_weights, dtype=float), (self.n_crossovers, n_comp))
        except ValueError as ex:
            raise e.ValueError(f"Inconsistent crossover parameters: {ex}") from ex
        if np.any(self.crossover_stds < 0):
            raise e.ValueError("`crossover_stds` should be non-negative")
        self.crossover_weights = self._normalize(weights, "crossover_weights")

        # Mutation: zero-mean Gaussian mixture of w_0 per block. Defaults: no
        # change, or a perturbation of 1% / 10% of the range; the probability of
        # the perturbation (1 / D) is set in `compile`
        self._default_mutation_weights = mutation_weights is None
        stds = (
            np.array([[0.0, 0.01], [0.0, 0.1]])[: self.n_mutations]
            if mutation_stds is None and self.n_mutations <= 2
            else np.asarray(mutation_stds if mutation_stds is not None else [0.0, 0.1], dtype=float)
        )
        stds = np.atleast_1d(stds)
        self.mutation_stds = np.broadcast_to(stds, (self.n_mutations, stds.shape[-1])).copy()
        if np.any(self.mutation_stds < 0):
            raise e.ValueError("`mutation_stds` should be non-negative")
        self.mutation_weights = (
            None
            if mutation_weights is None
            else self._normalize(
                np.broadcast_to(np.asarray(mutation_weights, dtype=float), self.mutation_stds.shape),
                "mutation_weights",
            )
        )

        # Ratio of each crossover block's offspring sent to each mutation block
        split = (
            np.array([0.4, 0.6])[: self.n_mutations]
            if crossover_split is None and self.n_mutations <= 2
            else np.asarray(crossover_split if crossover_split is not None else np.ones(self.n_mutations), dtype=float)
        )
        split = np.broadcast_to(split, (self.n_crossovers, self.n_mutations))
        self.crossover_split = self._normalize(split, "crossover_split")

    @staticmethod
    def _normalize(values: np.ndarray, name: str) -> np.ndarray:
        """Normalizes non-negative rows so that each one sums to one."""
        values = np.asarray(values, dtype=float)
        if np.any(values < 0) or np.any(values.sum(axis=-1) <= 0):
            raise e.ValueError(f"`{name}` should be non-negative with a positive sum per block")

        return values / values.sum(axis=-1, keepdims=True)

    def _graph_sizes(self, n_agents: int):
        """Parents per tournament block, so that every division is exact.

        Returns:
            (Tuple[int, int]): Parents per tournament block and offspring per generation.

        """

        target = n_agents if self.n_offspring is None else int(self.n_offspring)
        # Parents per tournament block must split equally among the exchange
        # blocks, and the exchange offspring equally among the crossover blocks
        per_tournament = self.n_exchanges * math.ceil(target / (self.n_tournaments * self.n_exchanges))
        while (self.n_tournaments * per_tournament // self.n_exchanges) % self.n_crossovers:
            per_tournament += self.n_exchanges

        return per_tournament, self.n_tournaments * per_tournament

    def _prepare(self, n_agents: int, n_variables: int, xp: Any, dtype: Any) -> Dict:
        """Builds the block parameters on the backend."""
        n_parents, n_offspring = self._graph_sizes(n_agents)

        if self._default_mutation_weights:
            rate = 1.0 / n_variables
            weights = np.zeros_like(self.mutation_stds)
            weights[:, 0], weights[:, 1:] = 1.0 - rate, rate / max(1, weights.shape[1] - 1)
            self.mutation_weights = weights

        # Tournament sizes as a mask over the largest tournament
        m_max = int(self.tournament_sizes.max())
        mask = np.arange(m_max)[None, None, :] < self.tournament_sizes[:, None, None]

        # Mutation block of every row of each crossover block (first rows to the
        # first mutation block, and so on, following `crossover_split`)
        rows = self.n_tournaments * n_parents // self.n_exchanges * self.n_exchanges // self.n_crossovers
        bounds = np.round(np.cumsum(self.crossover_split, axis=1) * rows).astype(int)
        routing = (np.arange(rows)[None, :, None] >= bounds[:, None, :]).sum(axis=2)
        routing = np.minimum(routing, self.n_mutations - 1)

        self.n_offspring_per_generation = n_offspring

        return {
            "n_tournaments": self.n_tournaments,
            "n_parents": n_parents,
            "m_max": m_max,
            "tournament_mask": xp.asarray(mask),
            "n_exchanges": self.n_exchanges,
            "exchange_size": self.exchange_size,
            "exchange_probabilities": xp.asarray(self.exchange_probabilities, dtype=dtype),
            "n_crossovers": self.n_crossovers,
            "crossover_size": self.crossover_size,
            "crossover_means": xp.asarray(self.crossover_means, dtype=dtype),
            "crossover_stds": xp.asarray(self.crossover_stds, dtype=dtype),
            "crossover_weights": xp.asarray(self.crossover_weights, dtype=dtype),
            "mutation_stds": xp.asarray(self.mutation_stds, dtype=dtype),
            "mutation_weights": xp.asarray(self.mutation_weights, dtype=dtype),
            "mutation_routing": xp.asarray(routing),
        }


_INIT_DOC = """
        Args:
            params: Contains key-value parameters to the meta-heuristics.
            n_tournaments: Number of Tournament blocks.
            n_exchanges: Number of Exchange blocks.
            n_crossovers: Number of Crossover blocks.
            n_mutations: Number of Mutation blocks.
            exchange_size: Parents per Exchange offspring (n_e).
            crossover_size: Parents per Crossover offspring (n_c).
            tournament_sizes: m of each Tournament block, in [1, 10].
            exchange_probabilities: (p_1, ..., p_ne) of each Exchange block.
            crossover_means: Means of the weights (w_2, ..., w_nc), of shape
                (n_crossovers, components, n_c - 1) (broadcastable).
            crossover_stds: Standard deviations with the same shape.
            crossover_weights: Probability of each mixture component per block.
            mutation_stds: Standard deviations of w_0 per block and component.
            mutation_weights: Probability of each mutation component per block.
                Defaults to a perturbation probability of 1 / n_variables.
            crossover_split: Ratio of each Crossover block's offspring sent to
                each Mutation block.
            n_offspring: Minimum number of offspring per generation (defaults to
                the population size); it is rounded up so that every division
                of the graph is exact.
"""


class AGGEA(_AGGEABase, Optimizer):
    """Automatic Generated Graph Evolutionary Algorithm (single-objective).

    The population flows through a graph of blocks: Tournament blocks select
    parents, Exchange blocks copy each variable from one of n_e parents, Crossover
    blocks combine n_c parents with Gaussian-mixture weights and Mutation blocks
    perturb the offspring; a (mu + lambda) Selection keeps the N best solutions.
    Every variation block is a translation, scale and rotation invariant operator.

    The report does not list the trained block parameters, so they are exposed
    as arguments; the defaults reproduce well-known operators (binary
    tournaments, uniform crossover, DE/rand/1/bin and Gaussian mutation).

    References:
        S. Zhu, H. Geng, X. Liu, Z. Zhu, R. Zhang, H. Jiang, C. He, and Y. Tian,
        "Report for the Automatic Generated Graph Evolutionary Algorithm,"
        GNBG competition report, 2025.

    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_tournaments: int = 10,
        n_exchanges: int = 10,
        n_crossovers: int = 2,
        n_mutations: int = 2,
        exchange_size: int = 2,
        crossover_size: int = 3,
        tournament_sizes: Optional[Sequence[int]] = None,
        exchange_probabilities: Optional[Any] = None,
        crossover_means: Optional[Any] = None,
        crossover_stds: Optional[Any] = None,
        crossover_weights: Optional[Any] = None,
        mutation_stds: Optional[Any] = None,
        mutation_weights: Optional[Any] = None,
        crossover_split: Optional[Any] = None,
        n_offspring: Optional[int] = None,
    ) -> None:
        logger.info("Overriding class: Optimizer -> AGGEA (Default).")

        super().__init__()

        self._set_blocks(
            n_tournaments, n_exchanges, n_crossovers, n_mutations, exchange_size,
            crossover_size, tournament_sizes, exchange_probabilities, crossover_means,
            crossover_stds, crossover_weights, mutation_stds, mutation_weights,
            crossover_split, n_offspring,
        )  # fmt: skip

        self.build(params)
        self._cfg = None

        logger.info("Class overrided.")

    __init__.__doc__ = "Initialization method.\n" + _INIT_DOC

    def compile(self, space: _SingleObjectiveSpace) -> None:
        """Prepares the block parameters for the space."""
        n_variables = space.agents[0].position.size
        self._cfg = self._prepare(space.n_agents, n_variables, np, float)

        self._lb = np.asarray(space.agents[0].lb, dtype=float).reshape(-1)
        self._ub = np.asarray(space.agents[0].ub, dtype=float).reshape(-1)

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
        """Performs one generation: offspring creation and (mu + lambda) selection.

        Args:
            space: A SingleObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        agents = space.agents
        shape = agents[0].position.shape
        X = np.stack([a.position.reshape(-1) for a in agents])
        F = np.array([float(np.asarray(a.fit).reshape(-1)[0]) for a in agents])

        O = _aggea_offspring(X, F, self._lb, self._ub, self._cfg, np)
        F_O = np.array([float(np.asarray(function(o.reshape(shape))).reshape(-1)[0]) for o in O])

        # Selection: the N best of the parents and the offspring
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


class AGGEATensor(_AGGEABase, Optimizer, TensorizedOptimizer):
    """Tensorized (NumPy / CuPy) AG-GEA: the same graph, run on the whole population.

    Every block is a batched gather, an element-wise operation or a sort, so a
    generation runs on the device without host synchronization.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_tournaments: int = 10,
        n_exchanges: int = 10,
        n_crossovers: int = 2,
        n_mutations: int = 2,
        exchange_size: int = 2,
        crossover_size: int = 3,
        tournament_sizes: Optional[Sequence[int]] = None,
        exchange_probabilities: Optional[Any] = None,
        crossover_means: Optional[Any] = None,
        crossover_stds: Optional[Any] = None,
        crossover_weights: Optional[Any] = None,
        mutation_stds: Optional[Any] = None,
        mutation_weights: Optional[Any] = None,
        crossover_split: Optional[Any] = None,
        n_offspring: Optional[int] = None,
    ) -> None:
        logger.info("Overriding class: Optimizer -> AGGEA (Tensor).")

        super().__init__()

        self._set_blocks(
            n_tournaments, n_exchanges, n_crossovers, n_mutations, exchange_size,
            crossover_size, tournament_sizes, exchange_probabilities, crossover_means,
            crossover_stds, crossover_weights, mutation_stds, mutation_weights,
            crossover_split, n_offspring,
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

        n_variables = int(np.prod(space.X.shape[1:]))
        self._cfg = self._prepare(space.n_agents, n_variables, xp, self.dtype)

        self._lb = xp.asarray(space.lb, dtype=self.dtype).reshape(-1)
        self._ub = xp.asarray(space.ub, dtype=self.dtype).reshape(-1)

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
        """Performs one generation: offspring creation and (mu + lambda) selection.

        Args:
            space: A tensorized SingleObjectiveSpace object.
            function: A Function object that will be used as the objective function.

        """

        xp = self._xp
        N = space.X.shape[0]
        feat_shape = space.X.shape[1:]

        X = space.X.reshape(N, -1)
        O = _aggea_offspring(X, space.F, self._lb, self._ub, self._cfg, xp).astype(self.dtype)
        F_O = function(O.reshape((O.shape[0],) + feat_shape), xp=xp).reshape(-1).astype(self.dtype)

        # Selection: the N best of the parents and the offspring, sorted
        F_all = xp.concatenate((space.F, F_O))
        best = xp.argsort(F_all)[:N]

        space.X = xp.concatenate((X, O))[best].reshape((N,) + feat_shape)
        space.F = F_all[best]
        self._update_best(space)
