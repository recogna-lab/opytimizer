"""MOEA/D."""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.utils.exception as e
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging
from opytimizer.utils.decomposition import tchebycheff
from opytimizer.utils.operators import PolynomialMutation, SBXCrossover

logger = logging.get_logger(__name__)


class MOEAD(Optimizer):
    """MOEAD class, inherited from Optimizer.

    References:
        Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
        IEEE Transactions on evolutionary computation, 11(6), 712-731.

    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        crossover_operator=None,
        mutation_operator=None,
        weights_vector=None,
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.

        """
        logger.info("Overriding class: Optimizer -> MOEAD.")

        super().__init__()

        self.CR = 1.0
        self.MR = 0.025
        self.n_subproblems = None
        self.neighborhood_size = None
        self.crossover_operator = crossover_operator or SBXCrossover()
        self.mutation_operator = mutation_operator or PolynomialMutation()
        self.weights_vector = weights_vector

        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_subproblems(self) -> int:
        """Number of subproblems."""
        return self._n_subproblems

    @n_subproblems.setter
    def n_subproblems(self, n_subproblems: Optional[int]) -> None:
        if n_subproblems is not None:
            if not isinstance(n_subproblems, int):
                raise e.TypeError("`n_subproblems` should be an integer.")
            if n_subproblems <= 0:
                raise e.ValueError("`n_subproblems` should be higher than 0.")

        self._n_subproblems = n_subproblems

    @property
    def CR(self) -> float:
        """Crossover probability."""
        return self._CR

    @CR.setter
    def CR(self, CR: float) -> None:
        if not isinstance(CR, (float, int)):
            raise e.TypeError("`CR` should be a float or integer.")
        if CR < 0 or CR > 1:
            raise e.ValueError("`CR` should be between 0 and 1.")

        self._CR = CR

    @property
    def MR(self) -> float:
        """Mutation rate."""
        return self._MR

    @MR.setter
    def MR(self, MR: float) -> None:
        if not isinstance(MR, (float, int)):
            raise e.TypeError("`MR` should be a float or integer.")
        if MR < 0 or MR > 1:
            raise e.ValueError("`MR` should be between 0 and 1.")

        self._MR = MR

    @property
    def neighborhood_size(self) -> int:
        """Size of neighborhood T."""
        return self._neighborhood_size

    @neighborhood_size.setter
    def neighborhood_size(self, neighborhood_size: Optional[int]) -> None:
        if neighborhood_size is not None:
            if not isinstance(neighborhood_size, int):
                raise e.TypeError("`neighborhood_size` should be an integer.")
            if neighborhood_size < 2:
                raise e.ValueError("`neighborhood_size` should be higher than 1.")

        self._neighborhood_size = neighborhood_size

    @property
    def T(self) -> np.ndarray:
        """Neighborhood of each subproblem."""
        return self._T

    @T.setter
    def T(self, T: np.ndarray) -> None:
        if not isinstance(T, np.ndarray):
            raise e.TypeError("`T` should be a numpy array.")
        self._T = T

    @property
    def weights_vector(self) -> np.ndarray:
        """Normalized weights vector."""
        return self._weights_vector

    @weights_vector.setter
    def weights_vector(self, weights: np.ndarray) -> None:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        self._weights_vector = weights

    @property
    def z(self) -> np.ndarray:
        """Reference point."""
        return self._z

    @z.setter
    def z(self, ref: np.ndarray) -> None:
        if not isinstance(ref, np.ndarray):
            ref = np.array(ref)
        self._z = ref

    def compile(self, space: Space, **kwargs) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """
        # If n_subproblems is not defined, use the number of agents
        if self.n_subproblems is None:
            self.n_subproblems = space.n_agents

        # If neighborhood_size is not defined, use 10% of the number of subproblems
        if self.neighborhood_size is None:
            self.neighborhood_size = max(2, int(self.n_subproblems * 0.1))

        # Build neighborhood
        self._build_neighborhood()

        # Initialize reference point
        self.z = np.full(space.n_objectives, np.inf)

        # Initialize iteration counter
        self._aux_iteration = 0

    def _build_neighborhood(self) -> None:
        """Builds the neighborhood using Euclidean distance."""
        n = len(self.weights_vector)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distances[i, j] = g.euclidean_distance(
                    self.weights_vector[i], self.weights_vector[j]
                )

        self.T = np.argsort(distances, axis=1)[:, : self.neighborhood_size]

    def _select_neighbors(self, index: int, space: Space) -> np.ndarray:
        """Selects neighbors for reproduction.

        Args:
            index: Current agent index.
            space: Space containing the population.

        Returns:
            (np.ndarray): Selected neighbor indices.

        """
        # Map the agent index to the subproblem index
        subproblem_idx = index % self.n_subproblems
        neighbors = self.T[subproblem_idx]

        # Ensure we have at least 2 neighbors
        if len(neighbors) < 2:
            return np.array([index, index])

        # Select 2 random different neighbors
        selected = np.random.choice(neighbors, 2, replace=False)

        # Map the subproblem indices back to the agent indices
        return np.array([idx % space.n_agents for idx in selected])

    def _genetic_operators(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Applies genetic operators.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            (tuple): Two generated children.

        """
        # Apply crossover respecting crossover_rate
        if np.random.random() < self.CR:
            child1, child2 = self.crossover_operator(parent1, parent2)
        else:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

        # Apply mutation respecting mutation_rate
        if np.random.random() < self.MR:
            child1 = self.mutation_operator(child1)
        if np.random.random() < self.MR:
            child2 = self.mutation_operator(child2)

        # Return the children
        return child1, child2

    def _update_neighborhood(
        self, index: int, new_agents: np.ndarray, new_f_values: np.ndarray, space: Space
    ) -> None:
        """Updates the population in the neighborhood.

        Args:
            index: Current agent index.
            new_agents: Newly generated agents.
            new_f_values: Fitness values of new agents.
            space: Space containing the population.

        """
        # Map the agent index to the subproblem index
        subproblem_idx = index % self.n_subproblems

        for i in self.T[subproblem_idx]:
            # Map the subproblem index to the agent index
            agent_idx = i % space.n_agents

            # Calculate fitness using decomposition
            fitness = np.array(
                [
                    tchebycheff(f_value, self.weights_vector[i], self.z)
                    for f_value in new_f_values
                ]
            )

            # Current fitness
            actual_f_value = space.agents[agent_idx].fit
            actual_fitness = tchebycheff(actual_f_value, self.weights_vector[i], self.z)

            # Update if better
            better_idx = np.argmin(fitness)
            if fitness[better_idx] <= actual_fitness:
                space.agents[agent_idx].position = copy.deepcopy(
                    new_agents[better_idx].position
                )
                space.agents[agent_idx].fit = copy.deepcopy(new_f_values[better_idx])

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the fitness of the agents.

        Args:
            space: Space containing agents and evaluation-related information.
            function: Function to evaluate the fitness of the agents.

        """
        # Initial evaluation (first iteration)
        if self._aux_iteration == 0:
            for agent in space.agents:
                agent.fit = function(agent.position)
                self.z = np.minimum(self.z, agent.fit)

            self._aux_iteration = 1

        # Update Pareto front
        space.update_pareto_front(space.agents)

    def update(self, space: Space, function: Function) -> None:
        """Updates the population using MOEA/D.

        Args:
            space: Space containing agents and update-related information.
            function: Function to evaluate the fitness of the agents.

        """
        # For each agent in the population
        for index, agent in enumerate(space.agents):
            # Select neighbors
            selected = self._select_neighbors(index, space)
            parent1 = space.agents[selected[0]]
            parent2 = space.agents[selected[1]]

            # Apply genetic operators
            offspring1, offspring2 = self._genetic_operators(parent1, parent2)

            # Evaluate new agents
            children = np.array([offspring1, offspring2])
            new_f_values = np.array([function(off.position) for off in children])

            # Update reference point
            self.z = np.minimum(self.z, np.min(new_f_values, axis=0))

            # Update population in neighborhood
            self._update_neighborhood(index, children, new_f_values, space)


class MOEAD_DE(Optimizer):
    """MOEA/D-DE class, inherited from Optimizer.

    References:
        Li, H., & Zhang, Q. (2008). Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II.
        IEEE transactions on evolutionary computation, 13(2), 284-302.

    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        mutation_operator=None,
        weights_vector=None,
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            mutation_operator: Mutation operator to be used.
            weights_vector: Weights vector to be used.

        """
        logger.info("Overriding class: Optimizer -> MOEAD_DE.")

        super().__init__()

        self.CR = 0.9
        self.MR = 0.05
        self.n_subproblems = None
        self.neighborhood_size = None
        self.mutation_operator = mutation_operator or PolynomialMutation()
        self.weights_vector = weights_vector
        self.nr = 2
        self.F = 0.5

        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_subproblems(self) -> int:
        """Number of subproblems."""
        return self._n_subproblems

    @n_subproblems.setter
    def n_subproblems(self, n_subproblems: Optional[int]) -> None:
        if n_subproblems is not None:
            if not isinstance(n_subproblems, int):
                raise e.TypeError("`n_subproblems` should be an integer.")
            if n_subproblems <= 0:
                raise e.ValueError("`n_subproblems` should be higher than 0.")

        self._n_subproblems = n_subproblems

    @property
    def CR(self) -> float:
        """Crossover probability."""
        return self._CR

    @CR.setter
    def CR(self, CR: float) -> None:
        if not isinstance(CR, (float, int)):
            raise e.TypeError("`CR` should be a float or integer.")
        if CR < 0 or CR > 1:
            raise e.ValueError("`CR` should be between 0 and 1.")

        self._CR = CR

    @property
    def MR(self) -> float:
        """Mutation rate."""
        return self._MR

    @MR.setter
    def MR(self, MR: float) -> None:
        if not isinstance(MR, (float, int)):
            raise e.TypeError("`MR` should be a float or integer.")
        if MR < 0 or MR > 1:
            raise e.ValueError("`MR` should be between 0 and 1.")

        self._MR = MR

    @property
    def nr(self) -> int:
        """Maximun number of replacements in the mating pool"""
        return self._nr

    @nr.setter
    def nr(self, nr: int) -> None:
        if not isinstance(nr, int):
            raise e.TypeError("`nr` should be an integer.")
        if nr < 1:
            raise e.ValueError("`nr` should be greater equal than 1.")
        self._nr = nr

    @property
    def neighborhood_size(self) -> int:
        """Size of neighborhood T."""
        return self._neighborhood_size

    @neighborhood_size.setter
    def neighborhood_size(self, neighborhood_size: Optional[int]) -> None:
        if neighborhood_size is not None:
            if not isinstance(neighborhood_size, int):
                raise e.TypeError("`neighborhood_size` should be an integer.")
            if neighborhood_size < 2:
                raise e.ValueError("`neighborhood_size` should be higher than 1.")

        self._neighborhood_size = neighborhood_size

    @property
    def T(self) -> np.ndarray:
        """Neighborhood of each subproblem."""
        return self._T

    @T.setter
    def T(self, T: np.ndarray) -> None:
        if not isinstance(T, np.ndarray):
            raise e.TypeError("`T` should be a numpy array.")
        self._T = T

    @property
    def weights_vector(self) -> np.ndarray:
        """Normalized weights vector."""
        return self._weights_vector

    @weights_vector.setter
    def weights_vector(self, weights: np.ndarray) -> None:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        self._weights_vector = weights

    @property
    def z(self) -> np.ndarray:
        """Reference point."""
        return self._z

    @z.setter
    def z(self, ref: np.ndarray) -> None:
        if not isinstance(ref, np.ndarray):
            ref = np.array(ref)
        self._z = ref

    @property
    def F(self) -> float:
        """The scalling factor"""
        return self._F

    @F.setter
    def F(self, F) -> None:
        if not isinstance(F, (float, int)):
            raise e.TypeError("`F` should be a float or an integer.")
        if F <= 0:
            raise e.ValueError("`F` should be greater than 0.")
        self._F = F

    def compile(self, space: Space, **kwargs) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """
        # If n_subproblems is not defined, use the number of agents
        if self.n_subproblems is None:
            self.n_subproblems = space.n_agents

        # If neighborhood_size is not defined, use 10% of the number of subproblems
        if self.neighborhood_size is None:
            self.neighborhood_size = max(2, int(self.n_subproblems * 0.1))

        # Build neighborhood
        self._build_neighborhood()

        # Initialize reference point
        self.z = np.full(space.n_objectives, np.inf)

        # Initialize iteration counter
        self._aux_iteration = 0

    def _build_neighborhood(self) -> None:
        """Builds the neighborhood using Euclidean distance."""
        n = len(self.weights_vector)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                distances[i, j] = g.euclidean_distance(
                    self.weights_vector[i], self.weights_vector[j]
                )

        self.T = np.argsort(distances, axis=1)[:, : self.neighborhood_size]

    def _select_neighbors(self, mating_pool: np.ndarray, space: Space) -> np.ndarray:
        """Selects neighbors for reproduction.

        Args:
            mating_pool: Current agent mating pool.
            space: Space containing the population.

        Returns:
            (np.ndarray): Selected neighbor indices.

        """

        # Ensure we have at least 2 neighbors
        if len(mating_pool) < 2:
            return np.array([mating_pool[0], mating_pool[0]])

        # Select 2 random different neighbors
        selected = np.random.choice(mating_pool, 3, replace=False)

        # Map the subproblem indices back to the agent indices
        return selected

    def _DE_operator(
        self,
        parent1: Agent,
        parent2: Agent,
        parent3: Agent,
        space: Space,
    ) -> np.ndarray:
        """Applies DE operator.
        Args:
            parent1: First parent.
            parent2: Second parent.
            parent3: Third parent
            space: Space containing boundary information

        Returns:
            np.ndarray: mutated solution position
        """

        # Apply DE
        child = copy.deepcopy(parent1)
        for i in range(len(parent1.position)):
            if np.random.rand() < self.CR:
                child.position[i] = parent1.position[i] + self.F * (
                    parent2.position[i] - parent3.position[i]
                )
                child.position[i] = np.clip(child.position[i], space.lb[i], space.ub[i])

        return child

    def _apply_operators(
        self,
        parent1: Agent,
        parent2: Agent,
        parent3: Agent,
        space: Space,
    ) -> tuple:
        """Applies DE and genetic operator.

        Args:
            parent1: First parent.
            parent2: Second parent.
            parent3: Third parent
            space: Space containing boundary information.

        Returns:
            np.ndarray: The generated children position.

        """

        # Apply DE operator
        child = self._DE_operator(
            parent1=parent1, parent2=parent2, parent3=parent3, space=space
        )

        # Apply mutation respecting mutation_rate
        if np.random.random() < self.MR:
            child = self.mutation_operator(child)

        return child

    def _update_neighborhood(
        self,
        new_agent: np.ndarray,
        new_f_value: np.ndarray,
        space: Space,
        mating_pool: np.ndarray,
    ) -> None:
        """Updates the population in the neighborhood.

        Args:
            new_agents: Newly generated agents.
            new_f_values: Fitness values of new agents.
            space: Space containing the population.
            mating_pool: Current agent mating pool.
        """

        c = 0

        while c < self.nr and len(mating_pool) > 0:

            i = np.random.choice(mating_pool, 1).item()

            # Calculate fitness using decomposition
            fitness = np.array(tchebycheff(new_f_value, self.weights_vector[i], self.z))

            # Current fitness
            actual_f_value = space.agents[i].fit
            actual_fitness = tchebycheff(actual_f_value, self.weights_vector[i], self.z)

            # Update if better

            if fitness <= actual_fitness:
                space.agents[i].position = copy.deepcopy(new_agent.position)
                space.agents[i].fit = copy.deepcopy(new_f_value)
                c += 1

            # Remove the index from mating pool

            idx = np.where(mating_pool == i)[0][0]
            mating_pool = np.delete(mating_pool, idx)

    def build_mating_pool(self, index) -> np.ndarray:
        """Build the mating of each agent"""
        """
        Args:
            index: current agent index.
        Returns:
            The agent mating pool
        """

        if np.random.rand() < 0.9:
            mating_pool = self.T[index].copy()
        else:
            mating_pool = np.array([i for i in range(self.n_subproblems)])

        return mating_pool

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the fitness of the agents.

        Args:
            space: Space containing agents and evaluation-related information.
            function: Function to evaluate the fitness of the agents.

        """
        # Initial evaluation (first iteration)
        if self._aux_iteration == 0:
            for agent in space.agents:
                agent.fit = function(agent.position)
                self.z = np.minimum(self.z, agent.fit)

            self._aux_iteration = 1

        # Update Pareto front
        space.update_pareto_front(space.agents)

    def update(self, space: Space, function: Function) -> None:
        """Updates the population using MOEA/D.

        Args:
            space: Space containing agents and update-related information.
            function: Function to evaluate the fitness of the agents.

        """
        # For each agent in the population
        for index, agent in enumerate(space.agents):

            # Build the mating pool
            mating_pool = self.build_mating_pool(index)

            # Select neighbors

            selected = self._select_neighbors(mating_pool, space)
            parent1 = space.agents[selected[0]]
            parent2 = space.agents[selected[1]]
            parent3 = space.agents[selected[2]]

            # Apply DE and genetic operator
            offspring = self._apply_operators(parent1, parent2, parent3, space)

            # Evaluate new agents
            new_f_value = function(offspring.position)

            # Update reference point
            self.z = np.minimum(self.z, new_f_value)

            # Update population in neighborhood
            self._update_neighborhood(offspring, new_f_value, space, mating_pool)
