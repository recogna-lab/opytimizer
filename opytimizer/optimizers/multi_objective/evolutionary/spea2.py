"""Strength Pareto Evolutionary Algorithm 2 (SPEA2)"""

import copy
from typing import List, Tuple

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function
from opytimizer.core.agent import Agent
from opytimizer.core.optimizer import MultiObjectiveOptimizer
from opytimizer.core.space import _MultiObjectiveSpace
from opytimizer.utils import logging
from opytimizer.utils.operators import PolynomialMutation, SBXCrossover

logger = logging.get_logger(__name__)


class SPEA2(MultiObjectiveOptimizer):
    """
    References:
        E. Zitzler, M. Laumanns, and L. Thiele. SPEA2: Improving the Strength Pareto
        Evolutionary Algorithm. Technical Report 103, TIK-Report, ETH Zurich (2001).
    """

    def __init__(
        self,
        params: dict = None,
        archive_size: int = 100,
        crossover_operator=None,
        mutation_operator=None,
    ) -> None:
        """Initialization method for the SPEA2 optimizer."""
        logger.info("Overriding class: MultiObjectiveOptimizer -> SPEA2.")

        super().__init__()

        self.archive_size = archive_size
        self.archive = []
        self.crossover_operator = crossover_operator or SBXCrossover(n_offspring=2)
        self.mutation_operator = mutation_operator or PolynomialMutation()

        self.build(params)

        logger.info("Class overrided.")

    @property
    def archive_size(self) -> int:
        return self._archive_size

    @archive_size.setter
    def archive_size(self, archive_size: int) -> None:
        if not isinstance(archive_size, int):
            raise e.TypeError("`archive_size` should be an integer")
        if archive_size < 0:
            raise e.ValueError("`archive_size` should be >= 0")

        self._archive_size = archive_size

    @property
    def archive(self) -> List[Agent]:
        return self._archive

    @archive.setter
    def archive(self, archive: List[Agent]) -> None:
        if not isinstance(archive, list):
            raise e.TypeError("`archive` should be a list")
        self._archive = archive

    @property
    def strength(self) -> np.ndarray:
        return self._strength

    @strength.setter
    def strength(self, strength: np.ndarray) -> None:
        if not isinstance(strength, np.ndarray):
            raise e.TypeError("`strength` should be a numpy array")
        self._strength = strength

    @property
    def raw_fitness(self) -> np.ndarray:
        return self._raw_fitness

    @raw_fitness.setter
    def raw_fitness(self, raw_fitness: np.ndarray) -> None:
        if not isinstance(raw_fitness, np.ndarray):
            raise e.TypeError("`raw_fitness` should be a numpy array")
        self._raw_fitness = raw_fitness

    @property
    def density(self) -> np.ndarray:
        return self._density

    @density.setter
    def density(self, density: np.ndarray) -> None:
        if not isinstance(density, np.ndarray):
            raise e.TypeError("`density` should be a numpy array")
        self._density = density

    def compile(self, space: _MultiObjectiveSpace) -> None:
        """Compiles additional information required for the optimizer."""
        self.archive = []
        self.strength = np.zeros(space.n_agents)
        self.raw_fitness = np.zeros(space.n_agents)
        self.density = np.zeros(space.n_agents)

    def _calculate_strength(self, dom_matrix: np.ndarray) -> None:
        """Calculates strength S(i) for each individual.

        Ref: Paper Sec. 3.1, Eq. 1 (p. 7)
        """
        # Eq. 1 (p. 7): S(i) = |{j | j in P_t + \bar{P}_t and i > j}|
        self.strength = np.sum(dom_matrix, axis=1, dtype=np.float64)

    def _calculate_raw_fitness(self, dom_matrix: np.ndarray) -> None:
        """Calculates raw fitness R(i) for each individual.

        Ref: Paper Sec. 3.1, Eq. 2 (p. 7)
        """
        # Eq. 2 (p. 7): R(i) = sum_{j in P_t + \bar{P}_t, j > i} S(j)
        self.raw_fitness = np.dot(dom_matrix.T.astype(np.float64), self.strength)

    def _calculate_density(self, agents: List[Agent]) -> None:
        """Calculates density estimate D(i) using the k-th nearest neighbor.

        Ref: Paper Sec. 3.1, Eq. 3 (p. 7)
        """
        n_agents = len(agents)
        if n_agents == 0:
            self.density = np.array([])
            return

        # k = sqrt(N + \bar{N}) (Sec. 3.1, p. 7)
        k = int(np.sqrt(n_agents))
        if k >= n_agents:
            k = n_agents - 1
        if k < 1:
            k = 1

        fits = np.atleast_2d(np.array([a.fit for a in agents]))
        if fits.ndim == 1:
            fits = fits[:, np.newaxis]

        # Euclidean distance matrix in objective space
        diff = fits[:, np.newaxis, :] - fits[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        # Sort distances of each agent in ascending order
        sorted_dists = np.sort(dists, axis=1)

        # Eq. 3 (p. 7): D(i) = 1 / (\sigma_i^k + 2)
        sigma_k = sorted_dists[:, k]
        self.density = 1.0 / (sigma_k + 2.0)

    def _update_metrics(self, agents: List[Agent]) -> None:
        """Updates S(i), R(i), and D(i) metrics for the given list of individuals."""
        n_agents = len(agents)
        if n_agents == 0:
            self.strength = np.array([])
            self.raw_fitness = np.array([])
            self.density = np.array([])
            return

        fits = np.atleast_2d(np.array([a.fit for a in agents]))
        if fits.ndim == 1:
            fits = fits[:, np.newaxis]

        # Vectorized Pareto Dominance Matrix (i > j)
        diff = fits[:, np.newaxis, :] - fits[np.newaxis, :, :]
        dom_matrix = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)

        # Eq. 1 (p. 7): Calculate S(i)
        self._calculate_strength(dom_matrix)
        # Eq. 2 (p. 7): Calculate R(i)
        self._calculate_raw_fitness(dom_matrix)
        # Eq. 3 (p. 7): Calculate D(i)
        self._calculate_density(agents)

    def _truncation(self, agents: List[Agent], target_size: int) -> List[Agent]:
        """Truncation operator: iteratively removes the individual with minimum
        distance to its neighbors in lexicographical order.

        Ref: Paper Sec. 3.2 (p. 8)
        """
        current_agents = list(agents)

        while len(current_agents) > target_size:
            L = len(current_agents)
            fits = np.atleast_2d(np.array([a.fit for a in current_agents]))
            if fits.ndim == 1:
                fits = fits[:, np.newaxis]

            diff = fits[:, np.newaxis, :] - fits[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)

            sorted_dists = np.sort(dists, axis=1)
            # Remove distance to self (index 0, which is 0.0)
            neighbor_dists = sorted_dists[:, 1:]

            # Sec. 3.2 (p. 8): Select agent with lexicographically minimum distance vector (i <=_d j)
            min_idx = min(range(L), key=lambda i: tuple(neighbor_dists[i]))
            current_agents.pop(min_idx)

        return current_agents

    def _environmental_selection(
        self, population: List[Agent], archive: List[Agent]
    ) -> List[Agent]:
        """Environmental Selection to maintain the external archive \bar{P}_{t+1}.

        Ref: Paper Sec. 3.2 (p. 8) & Algorithm 1 (Step 3, p. 5)
        """
        # Union of archive and population (P_t + \bar{P}_t)
        combined = population + archive
        self._update_metrics(combined)

        # Eq. 4 (p. 7): Total Fitness F(i) = R(i) + D(i)
        fitness = self.raw_fitness + self.density

        # Sec. 3.2 (p. 8): Copy non-dominated individuals (F(i) < 1, i.e., R(i) == 0)
        non_dominated = [
            agent for i, agent in enumerate(combined) if self.raw_fitness[i] == 0.0
        ]

        if len(non_dominated) == self.archive_size:
            return [copy.deepcopy(a) for a in non_dominated]
        elif len(non_dominated) < self.archive_size:
            # Sec. 3.2 (p. 8): Fill with best dominated individuals (lowest F(i) >= 1)
            dominated_indices = [
                i for i, _ in enumerate(combined) if self.raw_fitness[i] > 0.0
            ]
            dominated_indices.sort(key=lambda idx: fitness[idx])

            needed = self.archive_size - len(non_dominated)
            selected = list(non_dominated) + [
                combined[i] for i in dominated_indices[:needed]
            ]
            return [copy.deepcopy(a) for a in selected]
        else:
            # Sec. 3.2 (p. 8): Truncate non-dominated set to archive size
            truncated = self._truncation(non_dominated, self.archive_size)
            return [copy.deepcopy(a) for a in truncated]

    def _tournament_selection(self, agents: List[Agent], n_samples: int) -> List[Agent]:
        """Step 5 (Algorithm 1, p. 5): Binary tournament performed over archive \bar{P}_{t+1}."""
        if not agents:
            return []

        self._update_metrics(agents)
        # Eq. 4 (p. 7): F(i) = R(i) + D(i)
        fitness = self.raw_fitness + self.density
        n_agents = len(agents)
        selected = []

        for _ in range(n_samples):
            if n_agents == 1:
                selected.append(copy.deepcopy(agents[0]))
            else:
                i, j = np.random.choice(n_agents, 2, replace=False)
                winner = i if fitness[i] < fitness[j] else j
                selected.append(copy.deepcopy(agents[winner]))

        return selected

    def _crossover(self, parent1: Agent, parent2: Agent) -> Tuple:
        return self.crossover_operator(parent1, parent2)

    def _mutation(self, agent: Agent) -> Agent:
        return self.mutation_operator(agent)

    def _create_offspring(self, space: _MultiObjectiveSpace) -> List[Agent]:
        """Step 6 (Algorithm 1, p. 5): Variation (Crossover and Mutation on Mating Pool)."""
        n_offspring_needed = space.n_agents
        # Step 5 (Algorithm 1, p. 5): Selection of parents from archive
        mating_pool = self._tournament_selection(
            self.archive, n_samples=n_offspring_needed
        )

        offspring = []
        for i in range(0, len(mating_pool), 2):
            parent1 = mating_pool[i]
            parent2 = (
                mating_pool[i + 1] if (i + 1) < len(mating_pool) else mating_pool[0]
            )

            children = self._crossover(parent1, parent2)
            for child in children:
                mutated_child = self._mutation(child)
                offspring.extend(mutated_child)

        return offspring[:n_offspring_needed]

    def update(self, space: _MultiObjectiveSpace) -> None:
        """Executes one generation of the SPEA2 algorithm (Algorithm 1, p. 5)."""
        # Steps 2 & 3: Fitness Assignment & Environmental Selection to update archive \bar{P}_{t+1}
        self.archive = self._environmental_selection(space.agents, self.archive)

        # Steps 5 & 6: Mating Selection & Variation to generate new population P_{t+1}
        offspring = self._create_offspring(space)

        # Step 6: Update current space population with the new generation
        space.agents = offspring
