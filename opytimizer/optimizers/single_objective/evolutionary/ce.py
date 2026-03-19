"""
Chaotic Evolution and their variants 
"""
import copy
import numpy as np
import time

from typing import Optional, Dict, Any, Tuple
from typing_extensions import Literal, get_args
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging
import opytimizer.math.random as opt_r
logger = logging.get_logger(__name__)

_validSystems = Literal["logistic", "gauss", "tent", "henon"]

class CE(Optimizer):
    """
        A CE class, inherited from Optimizer.

        This is the designed class to define CE-related
        variables and methods.

        References:
            Pei, Y. (2020, October). Chaotic evolution algorithm with elite strategy in single-objective and multi-objective optimization.
            In 2020 IEEE international conference on systems, man, and cybernetics (SMC) (pp. 579-584). IEEE.

    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None, DR: float = 0.7, CR: float = 0.7, p: Tuple[float, float] = (0.02, 0.1), jump: int = 10, chaotic_system: _validSystems = 'tent'):
       """
        Initialization method
        Args:
            params: Contains key-value parameters to the meta-heuristics.
            `DR`: Direction rate.
            `CR`: Crossover rate.
            `jump`: Generation jump interval.
            `chaotic_system`: The chaotic system that will be adopted. The options are `'logistic'`, `'gauss'`, `'tent'` or `'henon'`.
            
       """
       super().__init__()
       
       self.DR = DR
       self.CR = CR
       self.jump = jump
       self.chaotic_system = chaotic_system
       
       self.cp = None
       self.D = None
       self.y_henon = None # henon case
       
       
       self.build(params)
       
       self.currentGen = 0
       
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
    def jump(self) -> int:
        return self._jump
    
    @jump.setter
    def jump(self, value: int) -> None:
        if not isinstance(value, int):
            raise e.TypeError('`jump` should be an intenger.')
        
        if value <= 0:
            raise e.ValueError('`jump` should be a positive value.')
        
        self._jump = value
        
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
       
    def compile(self, space):
        self.D = np.where(np.random.random((space.n_agents, space.n_variables)) < self.DR, -1.0, 1.0)
        self.cp = np.random.random((space.n_agents, space.n_variables))
        self.y_henon = np.random.random((space.n_agents, space.n_variables))
        self.chaotic_system_func = self._define_chaotic_system()
        
    def _logistic(self, x: np.ndarray, **kwargs) -> np.ndarray:
        u = 4.0
        x = u * x * (1 - x)
        return x
    
    def _tent(self, x: np.ndarray, **kw_args) -> np.ndarray:
        r = 2.0
        x = np.where(x < 0.5, r * x, r * (1 - x))
        return x
    
    def _gauss(self, x: np.ndarray, **kw_args) -> np.ndarray:
        a = 6.2
        b = -0.5
        x = np.exp(- a * (x ** 2) + b)
        return x
    
    def _henon(self, x: np.ndarray, i: int) -> np.ndarray:
        a = 1.4
        b = 0.3
        
        x_new = self.y_henon[i] - a * (x**2)
        self.y_henon[i] = b * x
        

        return np.clip(x_new, 0.0, 1.0)
    
    
    def evaluate(self, space, function):
      if self.currentGen == 0:
          super().evaluate(space, function)
          
      self.currentGen = self.currentGen + 1
        
        
        
    def update(self, space: Space, function: Function):
       chaotic_agents = copy.deepcopy(space.agents) 
       for i in range(space.n_agents):

           k = opt_r.generate_integer_random_number(low=0, high=space.n_variables)
           
           for j in range(space.n_variables):
               if (opt_r.generate_uniform_random_number() < self.CR) or (j == k): 
                    if self.currentGen % self.jump == 0:
                        chaotic_agents[i].position[j] = space.best_agent.position[j] * (1 + self.D[i][j] * self.cp[i][j])
                    else:
                        chaotic_agents[i].position[j] = chaotic_agents[i].position[j] * (1 + self.D[i][j] * self.cp[i][j]) 

       # repair
       for ag in chaotic_agents:
            ag.position = np.clip(ag.position.flatten(), ag.lb, ag.ub).reshape(-1,1)
            
        
        # selection
       for i in range(space.n_agents):
           chaotic_agents[i].fit = function(chaotic_agents[i].position)
           
           if chaotic_agents[i].fit < space.agents[i].fit:
               space.agents[i].fit = chaotic_agents[i].fit
               space.agents[i].position = chaotic_agents[i].position.copy()
               
               
               
       space.best_agent = copy.deepcopy(min(space.agents, key=lambda a: a.fit))
       space.best_agent.ts = int(time.time())
                   
       # chaotic_system
       for i in range(space.n_agents):
           self.cp[i] = self.chaotic_system_func(x=self.cp[i], i=i)
                   
            
                
           
       
       
       