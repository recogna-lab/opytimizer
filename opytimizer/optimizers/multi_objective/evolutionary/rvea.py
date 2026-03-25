"""Reference Vector Guided Evolutionary Algorithm"""

import numpy as np

from typing import Optional, Dict, Any, Union
import opytimizer.utils.exception as e
from opytimizer.core import MultiObjectiveOptimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.core.function import Function
from opytimizer.utils import logging
from opytimizer.utils.operators import SBXCrossover, PolynomialMutation
from opytimizer.utils.weights_vector import das_dennis

logger = logging.get_logger(__name__)


class RVEA(MultiObjectiveOptimizer):
    """
    RVEA class, inherited from MultiObjectiveOptimizer.

    References:
        R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization,"
        in IEEE Transactions on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, Oct. 2016, doi: 10.1109/TEVC.2016.2519378.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        crossover_operator=None,
        mutation_operator=None,
        reference_vectors: np.ndarray = None,
        max_generations: int = 250,
        alpha: Union[float, int] = 2.0,   
        fr: float = 0.1,      
    ):
        super().__init__()

        logger.info("Overriding class: MultiObjectiveOptimizer -> RVEA.")

        self.crossover_operator = crossover_operator or SBXCrossover(
            rate=1.0, gene_rate=1.0, return_mode="both"
        )
        self.mutation_operator = mutation_operator or PolynomialMutation(rate=1.0 / 30.0)
        self.reference_vectors = reference_vectors
        self.current_reference_vectors = reference_vectors.copy()
        self.max_generations = max_generations
        self.currentGeneration = 0
        self.z = None
        self.alpha = alpha
        self.fr = fr

        self.build(params)

    @property
    def max_generations(self) -> int:
        return self._max_generations
    @max_generations.setter
    def max_generations(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError('`max_generations` should be an integer.') 
        if value <= 0:
            raise e.ValueError('`max_generations` should be higher than 0.')
        
        self._max_generations = value
        
    @property
    def fr(self) -> float:
        return self._fr
    @fr.setter
    def fr(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise e.TypeError('`fr` should be an integer or a float.') 
        if value <= 0:
            raise e.ValueError('`fr` should be higher than 0.')
        
        self._fr = value
        
    @property
    def alpha(self) -> float:
        return self._alpha
    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isinstance(value, (float)):
            raise e.TypeError('`alpha` should be a float.') 
        if value <= 0:
            raise e.ValueError('`alpha` should be higher than 0.')
        
        self._alpha = value
        
    
           
    # ------------------------------------------------------------------
    def compile(self, space: Space):
        if len(self.reference_vectors) != space.n_agents:
            raise e.ValueError('Error: The number of `reference_vectors` should be equal to the number of agents.')
        
        #self.z = np.full(space.n_objectives, fill_value=np.inf)

    # ------------------------------------------------------------------
    def evaluate(self, space: Space, function: Function):
        if self.currentGeneration == 0:
            all_fits = []
            for agent in space.agents:
                agent.fit = function(agent.position)
                all_fits.append(agent.fit)
            all_fits = np.array(all_fits)
            self.z = np.min(all_fits, axis=0)
            print(self.z.shape)

        space.update_pareto_front(space.agents)

    # ------------------------------------------------------------------
    def _adapt_reference_vectors(self, space: Space):
        """Reference vector adaptation strategy — Algorithm 3, eq. (11).

        Adapts current_reference_vectors according to the objective value
        ranges so that solutions stay uniformly distributed even when
        objectives are scaled differently.
        """
        fr_period = max(1, int(self.max_generations * self.fr))
        if self.currentGeneration % fr_period != 0:
            return

        fits = np.array([agent.fit for agent in space.agents])  
        z_min = fits.min(axis=0)                                
        z_max = fits.max(axis=0)                                 

        scale = z_max - z_min
        scale = np.where(scale < 1e-10, 1e-10, scale)           

        # eq. (11):  v_{t+1,i} = (v0_i ⊙ scale) / ‖v0_i ⊙ scale‖
        adapted = self.reference_vectors * scale[None, :]        
        norms   = np.linalg.norm(adapted, axis=1, keepdims=True) 
        self.current_reference_vectors = adapted / norms

    # ------------------------------------------------------------------
    def update(self, space: Space, function: Function):

     
        num_pairs      = len(space.agents) // 2
        parent_indices = np.random.randint(0, len(space.agents), size=(num_pairs, 2))
        
      

        current_population = space.agents.copy()

        for pair in parent_indices:
            
            offsprings = self.crossover_operator(
                parent1=space.agents[pair[0]],
                parent2=space.agents[pair[1]],
            )
            for off in offsprings:
                self.mutation_operator(off)         
                off.fit = function(off.position)
                self.z = np.minimum(off.fit, self.z)
            current_population.extend(offsprings)

        N_pop = len(current_population)
        N_ref = len(self.current_reference_vectors)
        M     = space.n_objectives

        #  translation — eq. (5)
        # f'_{t,i} = f_{t,i} − z_min   
        translated = np.array(
            [agent.fit - self.z for agent in current_population]
        )                                              
        norms      = np.linalg.norm(translated, axis=1)     
        safe_norms = np.where(norms < 1e-10, 1e-10, norms)

        # Population partition — eq. (6) & (7) 
        # cos_matrix[i,j] = cos(θ) between individual i and ref vector j
        # Reference vectors are already unit vectors, so:
        #   cos θ = (f' / ‖f'‖) · v
        cos_matrix  = (translated / safe_norms[:, None]) @ self.current_reference_vectors.T
        cos_matrix  = np.clip(cos_matrix, -1.0, 1.0)       
        # each individual -> reference vector with maximum cosine (minimum angle)
        assignments = np.argmax(cos_matrix, axis=1)          

        # APD calculation — eq. (8), (9), (10)

        # γ_{v_j}: smallest angle between ref vector j and every other
        # ref vector  (eq. 10)
        ref_cos    = np.clip(
            self.current_reference_vectors @ self.current_reference_vectors.T,
            -1.0, 1.0,
        )                                                   
        ref_angles = np.arccos(ref_cos)                      
        np.fill_diagonal(ref_angles, np.inf)                 
        gammas = ref_angles.min(axis=1)                    

        # θ_{t,i,j}: angle between individual i and its assigned ref vector
        
        assigned_cosines = cos_matrix[np.arange(N_pop), assignments]  
        assigned_angles  = np.arccos(assigned_cosines)      
        
        # P(θ) — penalty function, eq. (9)
        t_ratio = self.currentGeneration / self.max_generations
        P = M * (t_ratio ** self.alpha) * (assigned_angles / gammas[assignments])  

        # APD — eq. (8):  d = (1 + P(θ)) · ‖f'‖
        apd_values = (1.0 + P) * norms                       

       
        # For each reference vector subpopulation, keep the individual
        # with the minimum APD value.
       
        new_agents = []
        for j in range(N_ref):
            mask = assignments == j
            if not mask.any():
                continue    # empty subpopulation
            indices  = np.where(mask)[0]
            best_idx = indices[np.argmin(apd_values[indices])]
            new_agents.append(current_population[best_idx])

        space.agents = new_agents
        
        self._adapt_reference_vectors(space)
        
        self.currentGeneration += 1