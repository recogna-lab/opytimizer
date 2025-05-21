"""
Multi Objective Evolutionary Algorithm based on Decomposition

"""

from typing import Any, Dict, Optional, Callable

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging
from scipy.spatial.distance import cdist
from opytimizer.utils.operators import sbx_crossover, polynomial_mutation
from opytimizer.utils.decomposition import weighted_sum
from opytimizer.utils.weights_vector import ref_dirs

logger = logging.get_logger(__name__)

class MOEAD(Optimizer):
    """
    References:
        Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
        IEEE Transactions on evolutionary computation, 11(6), 712-731.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """
        
        super().__init__()
        
        self.CR = 0.9
        self.MR = 0.05
        self.n_subproblems = 100
        self.neighborhood_size = int(self.n_subproblems/10)
        self.crossover_operator = sbx_crossover
        self.mutation_operator = polynomial_mutation
        self.decomposition = weighted_sum
        
        self.build(params)
        
        logger.info("Class overrided --> MOEA/D.")
    
    @property
    def n_problems(self) -> int:
        """Return the number of problems"""
        return self._n_problems

    @n_problems.setter
    def n_problems(self, n_problems: int) -> None:
        if not isinstance(n_problems, int):
            raise e.TypeError("`n_problems` should be an integer.")
        if n_problems < 2:
            raise e.ValueError("`The number of problems` should be higher equal than 2")
        self._n_problems = n_problems

    @property
    def n_subproblems(self) -> int:
        """Number of subproblems"""
        return self._n_subproblems
    
    @n_subproblems.setter
    def n_subproblems(self, n_subproblems: int) -> None:
        if not isinstance(n_subproblems, int):
            raise e.TypeError("`n_subproblems should be an integer.`")
        if n_subproblems <= 0:
            raise e.ValueError("`n_subproblems should be higher than 0.`")
        
        self._n_subproblems = n_subproblems
    
    @property
    def CR(self) -> float:
        """Crossover probability"""
        return self._CR
    
    @CR.setter
    def CR(self, CR: float) -> None:
        if not isinstance(CR, (float, int)):
            raise e.TypeError("`CR should be a float or integer.`")
        if CR < 0 or CR > 1:
            raise e.ValueError("`CR` should be between 0 and 1")
        
        self._CR = CR
        
    @property    
    def MR(self) -> float:
        """Mutation Rate"""
        return self._MR
    
    @MR.setter
    def MR(self, MR: float) -> None:
        if not isinstance(MR, (float, int)):
            raise e.TypeError("`MR` should be a float or integer.")
        if MR < 0 or MR > 1:
            raise e.ValueError("`MR` should be between 0 and 1.")
        
        self._MR = MR
        
    @property
    def crossover_operator(self) -> Callable:
        return self._crossover_operator
    
    @crossover_operator.setter
    def crossover_operator(self, op: Callable) -> None:
        self._crossover_operator = op
        
    @property
    def mutation_operator(self) -> Callable:
        return self._mutation_operator
    
    @mutation_operator.setter
    def mutation_operator(self, op: Callable) -> None:
        self._mutation_operator = op
        
    @property
    def neighborhood_size(self) -> int:
        """The size of neighborhood T"""
        return self._neighborhood_size
    
    @neighborhood_size.setter
    def neighborhood_size(self, neighborhood_size: int) -> None:
        if not isinstance(neighborhood_size, int):
            raise e.TypeError("`neighborhood_size` should be a intenger.`")
        if neighborhood_size < 2 or neighborhood_size > self.n_subproblems:
            raise e.ValueError(f"`neighborhood_size should be higher than 1 and less equal than {self.n_subproblems}")
        
        self._neighborhood_size = neighborhood_size
        
    @property
    def T(self) -> np.ndarray:
        """The neighborhood of each subproblem"""
        return self._T

    @T.setter
    def T(self, T: np.ndarray) -> None:
        if not isinstance(T, np.ndarray):
            raise e.ValueError("`T` should be a numpy array.")
        self._T = T
        
    @property
    def decomposition(self) -> float:
        """The decomposition function: calculates the fitness value"""
        return self._decomposition

    @decomposition.setter
    def decomposition(self, decomposition: Callable):
        if not isinstance(decomposition, Callable):
            raise e.TypeError("`decomposition` should be a 'Callable' type.")
        if decomposition.__module__ != 'opytimizer.utils.decomposition':
            raise e.TypeError("`decomposition` should be a valid function. Look in 'opytimizer.utils.decompostion' for valid functions.")
        
        self._decomposition = decomposition
      
    @property
    def weights_vector(self) -> np.ndarray:
        """The normalized weights vector"""  
        return self._weights_vector
    
    @weights_vector.setter
    def weights_vector(self, weights: np.ndarray) -> None:
        if not isinstance(weights, np.ndarray):
            weights = np.ndarray(weights)
        if weights.shape[0] != self.n_subproblems:
            raise e.ValueError("`weights_vector` number of rows should be exactly equal than `n_subproblems`.")
        self._weights_vector = weights
        
    @property
    def z(self) -> np.ndarray:
        """The reference point"""
        return self._z
    
    @z.setter
    def z(self, ref: np.ndarray) -> None:
        self._z = ref
        
    def compile(self, space: Space, **kwargs):
        self.n_problems = space.n_problems
        
        # Generate weight vectors
        self.weights_vector = ref_dirs(self.n_problems, self.n_subproblems)
        
        # Build neighborhood
        self._build_neighborhood()
        
        # Initialize reference point
        self.z = np.full(self.n_problems, np.inf)
        
        # Initialize fitness values
        for agent in space.agents:
            agent.fitness = np.zeros(self.n_problems)
        
        self._aux_iteration = 0
        
    def _build_neighborhood(self) -> None:
        """Computes the nearest neighbors for each weight vector using Euclidean distance."""
        distances = cdist(self.weights_vector, self.weights_vector, metric="euclidean")
        neighbors = np.argsort(distances, axis=1)[:, :self.neighborhood_size]
        
        self.T = neighbors
        
    def _genetic_operators(self, parent1: np.ndarray, parent2: np.ndarray, space: Space) -> np.ndarray:
        """Applies genetic operators (crossover and mutation) to generate offspring."""
        child1, child2 = self.crossover_operator(
            parent1, 
            parent2, 
            space.lower_bound, 
            space.upper_bound, 
            self.CR
        )
        
        child1 = self.mutation_operator(
            child1,
            space.lower_bound,
            space.upper_bound,
            self.MR
        )
        
        child2 = self.mutation_operator(
            child2,
            space.lower_bound,
            space.upper_bound,
            self.MR
        )
        
        return child1, child2
    
    def _select_neighbors(self, index: int) -> np.ndarray:
        """Selects two random neighbors from the neighborhood."""
        return np.random.choice(self.T[index], 2, replace=False)
    
    def _dominance_between_two_points(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """Checks if point1 dominates point2."""
        all_better = np.all(point1 <= point2)
        one_better = np.any(point1 < point2)
        return (all_better and one_better)
    
    def _update_population(self, new_agents: np.ndarray, index: int, functions: Function, space: Space) -> None:
        """Updates the population with new solutions."""
        new_f_values = np.zeros((2, self.n_problems))
        
        # Evaluate new solutions
        for i in range(len(new_f_values)):
            new_f_values[i] = functions(new_agents[i])
            self.z = np.minimum(self.z, new_f_values[i])
        
        # Update solutions in the neighborhood
        for i in self.T[index]:
            fitness = np.array([self.decomposition(f_value, self.weights_vector[i], self.z) for f_value in new_f_values])
            actual_f_value = space.agents[i].fitness
            actual_fitness = self.decomposition(actual_f_value, self.weights_vector[i], self.z)
            
            better_fitness_idx = np.argmin(fitness)
            
            if (fitness[better_fitness_idx] < actual_fitness or 
                (fitness[better_fitness_idx] == actual_fitness and 
                 self._dominance_between_two_points(new_f_values[better_fitness_idx], space.agents[i].fitness))):
                space.agents[i].position = new_agents[better_fitness_idx].copy()
                space.agents[i].fitness = new_f_values[better_fitness_idx].copy()
    
    def evaluate(self, space: Space, functions: Function):
        """Evaluates the population."""
        # Initialize the F_values
        if self._aux_iteration == 0:
            self._aux_iteration = 1
            for agent in space.agents:
                agent.fitness = functions(agent.position.flatten())
                self.z = np.minimum(self.z, agent.fitness)
        else:
            for index, agent in enumerate(space.agents):
                # Select neighbors
                selected_indexes = self._select_neighbors(index=index)
                parent1 = space.agents[selected_indexes[0]].position.ravel()
                parent2 = space.agents[selected_indexes[1]].position.ravel()
                
                # Generate offspring
                offspring1, offspring2 = self._genetic_operators(parent1, parent2, space)
                
                # Update population
                self._update_population(np.array([offspring1, offspring2]), index, functions, space) 