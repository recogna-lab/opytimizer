"""Language Education Optimization-based algorithms.
"""

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import _SingleObjectiveSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class LEO(Optimizer):
    """A LEO class, inherited from Optimizer.

    This is the designed class to define LEO-related
    variables and methods.

    References:
        P. Trojovsk`y, M. Dehghani, E. Trojovsk´a, and E. Milkova, “Language
        education optimization: A new human-based metaheuristic algorithm for
        solving optimization problems,” Computer Modeling in Engineering &
        Sciences, volume 136, issue: 2, 2023
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> LEO.")

        super(LEO, self).__init__()

        self.best_agent_index = 0 
            
        self.t = 1
        
        self.build(params)

        logger.info("Class overrided.")

    @property
    def local_position(self) -> np.ndarray:
        """Array of local positions."""

        return self._local_position

    @local_position.setter
    def local_position(self, local_position: np.ndarray) -> None:
        if not isinstance(local_position, np.ndarray):
            raise e.TypeError("`local_position` should be a numpy array")

        self._local_position = local_position

        
    def compile(self, space: _SingleObjectiveSpace) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        
    def evaluate(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit
                self.local_position[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())
                self.best_agent_index = i

        

    def update(self, space: _SingleObjectiveSpace, function: Function) -> None:
        """
        Args:
            space: Space containing a list of Agent objects, each with 'position' (NumPy array of shape (n, 1)) and 'fit' (scalar).
            function: A Function object that will be used as the objective function.
        """
        
        # To ensure compatibility with the position array shape (n, 1), the lower and upper bounds arrays are reshaped to (n, 1).
        reshaped_lb, reshaped_ub = space.lb.reshape(-1,1), space.ub.reshape(-1,1)

        # --- Phase 1: Teacher Selection and Learning ---

        # Array of fitness values for all agents        
        fitness = np.array([agent.fit for agent in space.agents])

        for i in range(space.n_agents):

            # Create mask to exclude the current agent and the best agent
            mask = (np.arange(space.n_agents) != i) & (np.arange(space.n_agents) != self.best_agent_index)

            # Select teachers: agents with better fitness than the current agent
            valid_teachers = np.where((fitness < fitness[i]) & mask)[0]
            
            if len(valid_teachers) == 0:
                selected_teacher = self.best_agent_index
            else:
                # Include the best agent as an option and select randomly
                valid_teachers = np.append(valid_teachers, self.best_agent_index)
                selected_teacher = np.random.choice(valid_teachers)
            
            # Vectorized position update
            r = np.random.rand() 
            I = np.random.randint(1, 3)
            new_pos = space.agents[i].position + r * (space.agents[selected_teacher].position - I * space.agents[i].position)

            # Apply bounds 
            new_pos = np.clip(new_pos, reshaped_lb, reshaped_ub)
            
            # Calculate new fitness
            new_fitness = function(new_pos)
            if new_fitness < space.agents[i].fit:
                space.agents[i].position = np.copy(new_pos)
                space.agents[i].fit = new_fitness

        # --- Phase 2: Student-to-Student Learning ---

            # Select a random student (different from the current agent)
            selected_student = np.random.randint(0, space.n_agents)
            while selected_student == i:
                selected_student = np.random.randint(0, space.n_agents)
            
            # Vectorized position update
            r = np.random.rand()  
            I = np.random.randint(1, 3)
            if space.agents[selected_student].fit < space.agents[i].fit:
                new_pos = space.agents[i].position + r * (space.agents[selected_student].position - I * space.agents[i].position)
            else:
                new_pos = space.agents[i].position + r * (space.agents[i].position - I * space.agents[selected_student].position)
            
            # Apply bounds
            new_pos = np.clip(new_pos, reshaped_lb, reshaped_ub)
            
            # Calculate new fitness
            new_fitness = function(new_pos)
            if new_fitness < space.agents[i].fit:
                space.agents[i].position = np.copy(new_pos)
                space.agents[i].fit = new_fitness

        # --- Phase 3: Individual Practice ---
            # Vectorized position update
            r = np.random.rand()  
            new_pos = space.agents[i].position + (reshaped_lb + r * (reshaped_ub - reshaped_lb)) / self.t
            
            # Apply bounds
            new_pos = np.clip(new_pos, reshaped_lb, reshaped_ub)
            
            # Calculate new fitness
            new_fitness = function(new_pos)
            if new_fitness < space.agents[i].fit:
                space.agents[i].position = np.copy(new_pos)
                space.agents[i].fit = new_fitness

        # Increment iteration counter
        self.t += 1