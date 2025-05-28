"""MOEA/D."""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space
from opytimizer.core.function import Function
from opytimizer.utils import logging
from opytimizer.utils.weights_vector import ref_dirs
from opytimizer.utils.operators import sbx_crossover, polynomial_mutation
from opytimizer.utils.decomposition import tchebycheff
from opytimizer.math.general import euclidean_distance

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
        crossover_params=None,
        mutation_params=None,
    ) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
            crossover_operator: Crossover operator to be used.
            mutation_operator: Mutation operator to be used.
            crossover_params: Parameters for the crossover operator.
            mutation_params: Parameters for the mutation operator.

        """
        logger.info("Overriding class: Optimizer -> MOEAD.")

        super().__init__()

        self.CR = 0.9
        self.MR = 0.05
        self.n_subproblems = None
        self.neighborhood_size = None
        self.crossover_operator = crossover_operator or sbx_crossover
        self.mutation_operator = mutation_operator or polynomial_mutation
        self.crossover_params = crossover_params or {}
        self.mutation_params = mutation_params or {}

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
        
        # Initialize weight vectors
        self.weights_vector = ref_dirs(
            n_objectives=space.n_objectives,
            n_partitions=self.n_subproblems
        )
        

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
                distances[i, j] = euclidean_distance(self.weights_vector[i], self.weights_vector[j])
        
       
        self.T = np.argsort(distances, axis=1)[:, :self.neighborhood_size]

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

    def _genetic_operators(self, parent1: np.ndarray, parent2: np.ndarray, space: Space) -> tuple:
        """Applies genetic operators.

        Args:
            parent1: First parent.
            parent2: Second parent.
            space: Space containing boundary information.

        Returns:
            (tuple): Two generated children.

        """
        # Apply crossover
        if self.crossover_operator == sbx_crossover:
            child1, child2 = self.crossover_operator(
                parent1=parent1,
                parent2=parent2,
                lb=space.lb,
                ub=space.ub,
                crossover_rate=self.CR,
                **self.crossover_params
            )
        else:
            child1, child2 = self.crossover_operator(
                parent1=parent1,
                parent2=parent2,
                lb=space.lb,
                ub=space.ub,
                **self.crossover_params
            )

        # Apply mutation
        if self.mutation_operator == polynomial_mutation:
            child1 = self.mutation_operator(
                vector=child1,
                lb=space.lb,
                ub=space.ub,
                mutation_rate=self.MR,
                **self.mutation_params
            )
            child2 = self.mutation_operator(
                vector=child2,
                lb=space.lb,
                ub=space.ub,
                mutation_rate=self.MR,
                **self.mutation_params
            )
        else:
            child1 = self.mutation_operator(
                vector=child1,
                lb=space.lb,
                ub=space.ub,
                **self.mutation_params
            )
            child2 = self.mutation_operator(
                vector=child2,
                lb=space.lb,
                ub=space.ub,
                **self.mutation_params
            )

        return child1, child2

    def _update_neighborhood(self, index: int, new_agents: np.ndarray, new_f_values: np.ndarray, space: Space) -> None:
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
            fitness = np.array([
                tchebycheff(f_value, self.weights_vector[i], self.z)
                for f_value in new_f_values
            ])
        
            # Current fitness
            actual_f_value = space.agents[agent_idx].fit.flatten()
            actual_fitness = tchebycheff(actual_f_value, self.weights_vector[i], self.z)

            # Update if better
            better_idx = np.argmin(fitness)
            if fitness[better_idx] <= actual_fitness:
                space.agents[agent_idx].position = new_agents[better_idx].copy()
                space.agents[agent_idx].fit = (new_f_values[better_idx].reshape(-1,1)).copy()

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
                self.z = np.minimum(self.z, agent.fit.T)
                
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
            parent1 = space.agents[selected[0]].position
            parent2 = space.agents[selected[1]].position

            # Apply genetic operators
            offspring1, offspring2 = self._genetic_operators(parent1, parent2, space)
            
            # Evaluate new agents
            new_agents = np.array([offspring1, offspring2])
            new_f_values = np.array([function(off).flatten() for off in new_agents])
            
            # Update reference point
            self.z = np.minimum(self.z, np.min(new_f_values, axis=0)).reshape(-1)
            

            # Update population in neighborhood
            self._update_neighborhood(index, new_agents, new_f_values, space) 