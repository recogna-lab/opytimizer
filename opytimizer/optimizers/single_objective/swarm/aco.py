"""Ant Colony Optimization for graph-structured search spaces.
"""

import copy
import time
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.graph.graph import Graph
from opytimizer.core.graph.node import GraphNode
from opytimizer.spaces.graph import _SingleObjectiveGraphSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class ACO(Optimizer):
    """An ACO class, inherited from Optimizer.

    References:
       M. Dorigo, V. Maniezzo and A. Colorni, "Ant system: optimization by a colony of cooperating agents,"
        in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics),
        vol. 26, no. 1, pp. 29-41, Feb. 1996, doi: 10.1109/3477.484436.

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> ACO.")

        super(ACO, self).__init__()

        self.alpha = 1.0
        self.beta = 5.0
        self.rho = 0.5
        self.q = 100.0
        self.tau_0 = 1.0

        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Relative importance of the pheromone trail."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Relative importance of the heuristic (visibility) information."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0:
            raise e.ValueError("`beta` should be >= 0")

        self._beta = beta

    @property
    def rho(self) -> float:
        """Pheromone persistence -- `(1 - rho)` is the evaporation rate."""

        return self._rho

    @rho.setter
    def rho(self, rho: float) -> None:
        if not isinstance(rho, (float, int)):
            raise e.TypeError("`rho` should be a float or integer")
        if rho < 0 or rho > 1:
            raise e.ValueError("`rho` should be between 0 and 1")

        self._rho = rho

    @property
    def q(self) -> float:
        """Constant related to the quantity of trail laid by each ant."""

        return self._q

    @q.setter
    def q(self, q: float) -> None:
        if not isinstance(q, (float, int)):
            raise e.TypeError("`q` should be a float or integer")
        if q < 0:
            raise e.ValueError("`q` should be >= 0")

        self._q = q

    @property
    def tau_0(self) -> float:
        """Initial pheromone level on every edge."""

        return self._tau_0

    @tau_0.setter
    def tau_0(self, tau_0: float) -> None:
        if not isinstance(tau_0, (float, int)):
            raise e.TypeError("`tau_0` should be a float or integer")
        if tau_0 < 0:
            raise e.ValueError("`tau_0` should be >= 0")

        self._tau_0 = tau_0

    @property
    def pheromone(self) -> np.ndarray:
        """`(n_nodes, n_nodes)` pheromone trail matrix."""

        return self._pheromone

    @pheromone.setter
    def pheromone(self, pheromone: np.ndarray) -> None:
        if not isinstance(pheromone, np.ndarray):
            raise e.TypeError("`pheromone` should be a numpy array")

        self._pheromone = pheromone

    @property
    def visibility(self) -> np.ndarray:
        """`(n_nodes, n_nodes)` heuristic desirability matrix. Defaults to
        all-ones (pure pheromone-driven search)."""

        return self._visibility

    @visibility.setter
    def visibility(self, visibility: np.ndarray) -> None:
        if not isinstance(visibility, np.ndarray):
            raise e.TypeError("`visibility` should be a numpy array")

        self._visibility = visibility

    def compile(self, space: _SingleObjectiveGraphSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A non-tensorized, single-objective GraphSpace object.

        """

        n_nodes = space.n_nodes

        self.pheromone = np.full((n_nodes, n_nodes), self.tau_0, dtype=float)
        self.visibility = np.ones((n_nodes, n_nodes), dtype=float)

    def evaluate(self, space: _SingleObjectiveGraphSpace, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A GraphSpace object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        for agent in space.agents:
            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = agent.position.copy()
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def _deposit_pheromone(self, space: _SingleObjectiveGraphSpace) -> None:
        """Evaporates old pheromone and deposits new trail proportional to
        each ant's solution quality (eqs. 1-3, adapted to minimization).

        Args:
            space: A GraphSpace object.

        """

        self.pheromone *= self.rho

        for agent in space.agents:
            deposit = self.q / (1.0 + agent.fit)

            node_index = {node: idx for idx, node in enumerate(agent.position.nodes)}
            for edge in agent.position.edges:
                i, j = node_index[edge.source], node_index[edge.target]

                self.pheromone[i, j] += deposit
                if not space.directed:
                    self.pheromone[j, i] += deposit

    def _construct_graph(self, space: _SingleObjectiveGraphSpace) -> Graph:
        """Builds one new candidate `Graph`: every possible edge is kept
        or dropped independently, with probability driven by pheromone x
        visibility (the generalization of eq. 4 described in this class'
        docstring).

        Args:
            space: A GraphSpace object.

        """

        n_nodes = space.n_nodes

        graph = Graph(space.directed)
        nodes = [GraphNode(name=f"n{i}") for i in range(n_nodes)]
        for node in nodes:
            graph.add_node(node)

        for i in range(n_nodes):
            j_range = range(n_nodes) if space.directed else range(i + 1, n_nodes)
            for j in j_range:
                if i == j:
                    continue

                desirability = (self.pheromone[i, j] ** self.alpha) * (
                    self.visibility[i, j] ** self.beta
                )
                probability = desirability / (1.0 + desirability)

                if r.generate_uniform_random_number() < probability:
                    graph.add_edge(nodes[i], nodes[j])

        return graph

    def update(self, space: _SingleObjectiveGraphSpace) -> None:
        """Wraps Ant System over the whole colony: evaporates/deposits
        pheromone based on the --current-- generation's graphs, then has
        every ant construct a new graph for the next one.

        Args:
            space: GraphSpace containing agents and update-related information.

        """

        self._deposit_pheromone(space)

        for agent in space.agents:
            agent.position = self._construct_graph(space)


class TSPACO(ACO):
    """Ant Colony Optimization specialized for the Traveling Salesman Problem.

    Each ant constructs a Hamiltonian cycle by selecting the next
    unvisited city according to pheromone and heuristic information.

    Args:
        params: Contains key-value parameters to the meta-heuristic.
        distance_matrix: Matrix containing pairwise distances between cities.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        distance_matrix: Optional[np.ndarray] = None,
    ) -> None:

        logger.info("Overriding class: ACO -> TSPACO.")

        super().__init__(params)

        if distance_matrix is None:
            raise e.ValueError("`distance_matrix` should not be None")

        if not isinstance(distance_matrix, np.ndarray):
            raise e.TypeError("`distance_matrix` should be a numpy array")

        if distance_matrix.ndim != 2:
            raise e.ValueError("`distance_matrix` should be a 2D array")

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise e.ValueError("`distance_matrix` should be square")

        self.distance_matrix = distance_matrix

        logger.info("Class overrided.")

    def compile(
        self,
        space: _SingleObjectiveGraphSpace,
    ) -> None:
        """Compiles pheromone and TSP heuristic information."""

        super().compile(space)

        n_nodes = space.n_nodes

        if self.distance_matrix.shape != (n_nodes, n_nodes):
            raise e.ValueError(
                "`distance_matrix` shape should match the number of nodes"
            )

        # Standard TSP heuristic:
        #
        # eta(i,j) = 1 / distance(i,j)
        #
        # Shorter edges are therefore more desirable.

        self.visibility = np.zeros(
            (n_nodes, n_nodes),
            dtype=float,
        )

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and self.distance_matrix[i, j] > 0:
                    self.visibility[i, j] = 1.0 / self.distance_matrix[i, j]

    def _construct_graph(
        self,
        space: _SingleObjectiveGraphSpace,
    ) -> Graph:
        """Constructs one valid Hamiltonian cycle."""

        n_nodes = space.n_nodes

        graph = Graph(directed=True)

        nodes = [GraphNode(name=f"n{i}") for i in range(n_nodes)]

        for node in nodes:
            graph.add_node(node)

        # Start from the first city.
        current = 0

        # City 0 is already visited.
        unvisited = set(range(1, n_nodes))

        while unvisited:

            candidates = list(unvisited)

            desirabilities = np.array(
                [
                    (self.pheromone[current, j] ** self.alpha)
                    * (self.visibility[current, j] ** self.beta)
                    for j in candidates
                ],
                dtype=float,
            )

            total = np.sum(desirabilities)

            # Numerical safety fallback.
            if total <= 0 or not np.isfinite(total):
                probabilities = np.ones(
                    len(candidates),
                    dtype=float,
                )

                probabilities /= probabilities.sum()

            else:
                probabilities = desirabilities / total

            # Roulette-wheel selection.
            next_city = np.random.choice(
                candidates,
                p=probabilities,
            )

            graph.add_edge(
                nodes[current],
                nodes[next_city],
            )

            unvisited.remove(next_city)

            current = next_city

        # Close the Hamiltonian cycle.
        graph.add_edge(
            nodes[current],
            nodes[0],
        )

        return graph

    def _deposit_pheromone(
        self,
        space: _SingleObjectiveGraphSpace,
    ) -> None:
        """Evaporates pheromone and deposits Q / tour_length."""

        self.pheromone *= self.rho

        for agent in space.agents:

            # Invalid solutions are not expected because
            # _construct_graph() always builds a Hamiltonian cycle.
            if agent.fit <= 0:
                continue

            deposit = self.q / agent.fit

            node_index = {node: idx for idx, node in enumerate(agent.position.nodes)}

            for edge in agent.position.edges:

                i = node_index[edge.source]
                j = node_index[edge.target]

                self.pheromone[i, j] += deposit
