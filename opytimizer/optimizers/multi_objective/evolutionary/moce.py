"""
Multi-Objective Chaotic Evolution Algorithms
"""
import copy
import numpy as np

from typing import Optional, Dict, Any, List
from typing_extensions import Literal, get_args
from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging
import opytimizer.utils.exception as e

logger = logging.get_logger(__name__)

_validSystems = Literal["logistic", "gauss", "tent", "henon"]


class MOCE(MultiObjectiveOptimizer):
    """
    Multi-Objective Chaotic Evolution (MOCE).

    Uses chaotic ergodicity combined with non-dominated sorting
    and crowding distance selection (NSGA-II style) for
    multi-objective optimization.

    References:
        Y. Pei, "Chaotic Evolution Algorithm with Elite Strategy
        in Single-objective and Multi-objective Optimization,"
        2020 IEEE International Conference on Systems, Man,
        and Cybernetics (SMC), Toronto, Canada, 2020,
        pp. 579-584.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        DR: float = 0.7,
        CR: float = 0.7,
        chaotic_system: _validSystems = 'logistic',
    ):
        """
        Initialization method.

        Args:
            params    : Key-value parameters for the meta-heuristic.
            `DR`        : Direction factor rate.
            `CR`        : Crossover rate.
            `chaotic_system`: Chaotic map to use. Options: `'logistic'`,
                            `'gauss'`, `'tent'`, `'henon'`.
        """
        super().__init__()

        self.DR = DR
        self.CR = CR
        self.chaotic_system = chaotic_system

        self.cp = None
        self.D = None
        self.y_henon = None  # only used by henon map
        self.currentGen = 0

        self.build(params)

        logger.info("Class overrided.")



    @property
    def DR(self) -> float:
        return self._DR
    
    @DR.setter
    def DR(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('`DR` should be a float.')
        
        if value < 0.0 or value > 1.0:
            raise e.ValueError('`DR` should be within [0.0, 1.0] interval.')
        
        self._DR = value
        
    @property
    def CR(self) -> float:
        return self._CR
    
    @CR.setter
    def CR(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('`CR` should be a float.')
        
        if value < 0.0 or value > 1.0:
            raise e.ValueError('`CR` should be within [0.0, 1.0] interval.')
        
        self._CR = value
        
    @property
    def chaotic_system(self) -> str:
        return self._chaotic_system
    
    @chaotic_system.setter
    def chaotic_system(self, value: _validSystems) -> None:
        if value not in get_args(_validSystems):
            raise e.ValueError(f'`chaotic_system` possible values are: {get_args(_validSystems)}')
        
        self._chaotic_system = value
        
        
    def _define_chaotic_system(self):
        return getattr(self, f'_{self.chaotic_system}')

    def compile(self, space: Space):
        self.D = np.where(
            np.random.random((space.n_agents, space.n_variables)) < self.DR,
            -1.0, 1.0
        )
        self.cp = np.random.random((space.n_agents, space.n_variables))
        self.y_henon = np.random.random((space.n_agents, space.n_variables))
        self.chaotic_system_func = self._define_chaotic_system()

    # ------------------------------------------------------------------
    # Chaotic systems
    # ------------------------------------------------------------------
    
    def _logistic(self, x: np.ndarray, **kwargs) -> np.ndarray:
        u = 4.0
        return u * x * (1 - x)

    def _tent(self, x: np.ndarray, **kwargs) -> np.ndarray:
        r = 2.0
        return np.where(x < 0.5, r * x, r * (1 - x))

    def _gauss(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a, b = 6.2, -0.5
        return np.exp(-a * (x ** 2) + b)

    def _henon(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a, b = 1.4, 0.3
        x_new = self.y_henon - a * (x ** 2)
        self.y_henon = b * x
        return x_new

    # ------------------------------------------------------------------
    # Pareto utilities
    # ------------------------------------------------------------------

    def _non_dominated_sort(self, fits: np.ndarray) -> List[List[int]]:
        """
       

        Args:
            fits: Array of shape (n, n_objectives).

        Returns:
            List of Pareto fronts, each front is a list of agent indices.
        """
        n = len(fits)

        # vectorized dominance matrix: dom[i,j] = True if i dominates j
        fits_i = fits[:, np.newaxis, :]   # (n, 1, n_obj)
        fits_j = fits[np.newaxis, :, :]   # (1, n, n_obj)
        dom = (
            np.all(fits_i <= fits_j, axis=2) &
            np.any(fits_i < fits_j, axis=2)
        )  # (n, n)
        np.fill_diagonal(dom, False)

        dominated_count = dom.sum(axis=0).astype(int)
        dominates_set = [list(np.where(dom[i])[0]) for i in range(n)]

        fronts = [list(np.where(dominated_count == 0)[0])]

        current = 0
        while fronts[current]:
            next_front = []
            for i in fronts[current]:
                for j in dominates_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            current += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _crowding_distance(self, fits: np.ndarray, front: List[int]) -> np.ndarray:
        """
        Computes crowding distance for individuals in a front.

        Args:
            fits : Array of shape (n_agents, n_objectives).
            front: List of indices belonging to this front.

        Returns:
            Array of crowding distances for each index in front.
        """
        n = len(front)
        distances = np.zeros(n)

        if n <= 2:
            distances[:] = np.inf
            return distances

        n_obj = fits.shape[1]
        for m in range(n_obj):
            obj_vals = fits[front, m]
            order = np.argsort(obj_vals)
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

            obj_range = obj_vals[order[-1]] - obj_vals[order[0]]
            if obj_range == 0:
                continue

            for k in range(1, n - 1):
                distances[order[k]] += (
                    obj_vals[order[k + 1]] - obj_vals[order[k - 1]]
                ) / obj_range

        return distances


    def evaluate(self, space: Space, function: Function):
        if self.currentGen == 0:
            for ag in space.agents:
                ag.fit = function(ag.position)

        self.currentGen += 1
        space.update_pareto_front(space.agents)

    def update(self, space: Space, function: Function):
     
        chaotic_agents = copy.deepcopy(space.agents)

        # crossover mask: shape (n_agents, n_variables)
        rand_matrix = np.random.random((space.n_agents, space.n_variables))
        k_indices = np.random.randint(0, space.n_variables, size=space.n_agents)
        k_mask = np.zeros((space.n_agents, space.n_variables), dtype=bool)
        k_mask[np.arange(space.n_agents), k_indices] = True
        cross_mask = (rand_matrix < self.CR) | k_mask  # (n_agents, n_variables)

        # positions matrix: shape (n_agents, n_variables)
        positions = np.array([ag.position.flatten() for ag in chaotic_agents])
        mutant = positions * (1 + self.D * self.cp)
        positions = np.where(cross_mask, mutant, positions)

        for i, ag in enumerate(chaotic_agents):
            ag.position = positions[i].reshape(-1, 1)

  
        for ag in chaotic_agents:
            ag.position = np.clip(
                ag.position.flatten(), ag.lb, ag.ub
            ).reshape(-1, 1)

       
        for ag in chaotic_agents:
            ag.fit = function(ag.position)

        
        pool = space.agents + chaotic_agents  # 2 * PS individuals

        fits = np.array([np.atleast_1d(ag.fit) for ag in pool])  # (2*PS, n_obj)

        
        fronts = self._non_dominated_sort(fits)

      
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= space.n_agents:
                selected.extend(front)
            else:
                needed = space.n_agents - len(selected)
                distances = self._crowding_distance(fits, front)
                order = np.argsort(-distances)
                selected.extend([front[o] for o in order[:needed]])
                break

       
        for idx, pool_idx in enumerate(selected):
            space.agents[idx] = copy.deepcopy(pool[pool_idx])

        
        self.D = np.where(
            np.random.random((space.n_agents, space.n_variables)) < self.DR,
            -1.0, 1.0
        )
        self.cp = self.chaotic_system_func(self.cp)
        
        
        
        
class OBMOCE(MultiObjectiveOptimizer):
    """
    Opposition Learning-based Multi-Objective Chaotic Evolution (OBMOCE).

    Extends MOCE by integrating the OBL mechanism (CEcOB variant):
    for each individual, generates a chaotic vector and its opposite,
    then selects the next generation from the combined pool of
    current population + chaotic + opposite using non-dominated
    sorting and crowding distance.

    Pool size per generation: 3 x PS.

    References:
       Li, T., & Pei, Y. (2025). Opposition-based chaotic evolution for optimization.
       Scientific Reports, 15(1), 22718.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        DR: float = 0.5,
        CR: float = 0.9,
        chaotic_system: _validSystems = 'logistic',
    ):
        """
        Initialization method.

        Args:
            params          : Key-value parameters for the meta-heuristic.
            `DR`            : Direction factor rate (paper default: 0.5).
            `CR`            : Crossover rate (paper default: 0.9).
            `chaotic_system`: Chaotic map. Options: `'logistic'`, `'gauss'`,
                              `'tent'`, `'henon'`. Paper uses `'logistic'`.
        """
        super().__init__()

        self.DR = DR
        self.CR = CR
        self.chaotic_system = chaotic_system

        self.cp = None
        self.D = None
        self.y_henon = None
        self.currentGen = 0

        self.build(params)
        logger.info("Class overrided.")
        
        
    @property
    def DR(self) -> float:
        return self._DR
    
    @DR.setter
    def DR(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('`DR` should be a float.')
        
        if value < 0.0 or value > 1.0:
            raise e.ValueError('`DR` should be within [0.0, 1.0] interval.')
        
        self._DR = value
        
    @property
    def CR(self) -> float:
        return self._CR
    
    @CR.setter
    def CR(self, value: float) -> None:
        if not isinstance(value, float):
            raise e.TypeError('`CR` should be a float.')
        
        if value < 0.0 or value > 1.0:
            raise e.ValueError('`CR` should be within [0.0, 1.0] interval.')
        
        self._CR = value
        
    
    @property
    def chaotic_system(self) -> str:
        return self._chaotic_system
    
    @chaotic_system.setter
    def chaotic_system(self, value: _validSystems) -> None:
        if value not in get_args(_validSystems):
            raise e.ValueError(f'`chaotic_system` possible values are: {get_args(_validSystems)}')
        
        self._chaotic_system = value

    # ------------------------------------------------------------------
    # Chaotic systems
    # ------------------------------------------------------------------

    def _define_chaotic_system(self):
        return getattr(self, f'_{self.chaotic_system}')

    def compile(self, space: Space):
        self.D = np.where(
            np.random.random((space.n_agents, space.n_variables)) < self.DR,
            -1.0, 1.0
        )
        self.cp = np.random.random((space.n_agents, space.n_variables))
        self.y_henon = np.random.random((space.n_agents, space.n_variables))
        self.chaotic_system_func = self._define_chaotic_system()

    def _logistic(self, x: np.ndarray) -> np.ndarray:
        return 4.0 * x * (1 - x)

    def _tent(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.5, 2.0 * x, 2.0 * (1 - x))

    def _gauss(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-6.2 * (x ** 2) - 0.5)

    def _henon(self, x: np.ndarray) -> np.ndarray:
        x_new = self.y_henon - 1.4 * (x ** 2)
        self.y_henon = 0.3 * x
        return x_new

    # ------------------------------------------------------------------
    # OBL utility
    # ------------------------------------------------------------------

    def _opposite(
        self,
        positions: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """
        Compute opposite population (Eq. 8):
            OP(x_j) = a_j + b_j - x_j

        Args:
            positions : (n_agents, n_variables)
            lb        : (n_variables,)
            ub        : (n_variables,)

        Returns:
            Opposite positions (n_agents, n_variables), within [lb, ub].
        """
        return lb + ub - positions

    # ------------------------------------------------------------------
    # Pareto utilities
    # ------------------------------------------------------------------

    def _non_dominated_sort(self, fits: np.ndarray) -> List[List[int]]:
        """
        Non-dominated sorting.

        Args:
            fits: (n, n_objectives)

        Returns:
            List of Pareto fronts (list of index lists).
        """
        n = len(fits)

        fits_i = fits[:, np.newaxis, :]   # (n, 1, n_obj)
        fits_j = fits[np.newaxis, :, :]   # (1, n, n_obj)
        dom = (
            np.all(fits_i <= fits_j, axis=2) &
            np.any(fits_i < fits_j, axis=2)
        )  # dom[i,j] = True if i dominates j
        np.fill_diagonal(dom, False)

        dominated_count = dom.sum(axis=0).astype(int)
        dominates_set = [list(np.where(dom[i])[0]) for i in range(n)]

        fronts = [list(np.where(dominated_count == 0)[0])]

        current = 0
        while fronts[current]:
            next_front = []
            for i in fronts[current]:
                for j in dominates_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            current += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _crowding_distance(self, fits: np.ndarray, front: List[int]) -> np.ndarray:
        """
        Crowding distance for individuals in a front.

        Args:
            fits : (n_pool, n_objectives)
            front: List of pool indices in this front.

        Returns:
            Crowding distances for each index in front.
        """
        n = len(front)
        distances = np.zeros(n)

        if n <= 2:
            distances[:] = np.inf
            return distances

        n_obj = fits.shape[1]
        for m in range(n_obj):
            obj_vals = fits[front, m]
            order = np.argsort(obj_vals)
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

            obj_range = obj_vals[order[-1]] - obj_vals[order[0]]
            if obj_range == 0:
                continue

            for k in range(1, n - 1):
                distances[order[k]] += (
                    obj_vals[order[k + 1]] - obj_vals[order[k - 1]]
                ) / obj_range

        return distances

  
    def evaluate(self, space: Space, function: Function):
        if self.currentGen == 0:
            for ag in space.agents:
                ag.fit = function(ag.position)

        self.currentGen += 1
        space.update_pareto_front(space.agents)

    def update(self, space: Space, function: Function):
      
        lb = space.agents[0].lb
        ub = space.agents[0].ub

        # Generate chaotic vectors 
        rand_matrix = np.random.random((space.n_agents, space.n_variables))
        k_indices = np.random.randint(0, space.n_variables, size=space.n_agents)
        k_mask = np.zeros((space.n_agents, space.n_variables), dtype=bool)
        k_mask[np.arange(space.n_agents), k_indices] = True
        cross_mask = (rand_matrix < self.CR) | k_mask  # (n_agents, n_variables)

        target_pos = np.array([ag.position.flatten() for ag in space.agents])
        mutant = target_pos * (1 + self.D * self.cp)
        chaotic_pos = np.where(cross_mask, mutant, target_pos)

        # Repair chaotic vectors 
        chaotic_pos = np.clip(chaotic_pos, lb, ub)

        # Generate opposite of chaotic vectors (OBL, Eq. 8)
        # OP(c_j) = lb_j + ub_j - c_j — already within [lb, ub]
        opposite_pos = self._opposite(chaotic_pos, lb, ub)

        # Build chaotic and opposite agent lists 
        chaotic_agents = copy.deepcopy(space.agents)
        opposite_agents = copy.deepcopy(space.agents)

        for i in range(space.n_agents):
            chaotic_agents[i].position = chaotic_pos[i].reshape(-1, 1)
            opposite_agents[i].position = opposite_pos[i].reshape(-1, 1)

        # Evaluate chaotic and opposite populations 
        for ag in chaotic_agents:
            ag.fit = function(ag.position)

        for ag in opposite_agents:
            ag.fit = function(ag.position)

        # Pool = current (PS) + chaotic (PS) + opposite (PS) 
        pool = space.agents + chaotic_agents + opposite_agents  # 3 × PS

        fits = np.array([np.atleast_1d(ag.fit) for ag in pool])  # (3*PS, n_obj)

        # Non-dominated sorting 
        fronts = self._non_dominated_sort(fits)

        # Select PS individuals via crowding distance 
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= space.n_agents:
                selected.extend(front)
            else:
                needed = space.n_agents - len(selected)
                distances = self._crowding_distance(fits, front)
                order = np.argsort(-distances)
                selected.extend([front[o] for o in order[:needed]])
                break

        # Update population 
        for idx, pool_idx in enumerate(selected):
            space.agents[idx] = copy.deepcopy(pool[pool_idx])


        #  Update D and CP 
        self.D = np.where(
            np.random.random((space.n_agents, space.n_variables)) < self.DR,
            -1.0, 1.0
        )
        self.cp = self.chaotic_system_func(self.cp)