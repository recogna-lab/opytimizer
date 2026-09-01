"""Optimizer.
"""

import copy
import time
from typing import Any, Dict, Union
import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.space import (_SingleObjectiveSpace, _MultiObjectiveSpace,
                                   _SingleObjectiveTensorSpace, _MultiObjectiveTensorSpace)

from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Optimizer:
    """An Optimizer class that holds meta-heuristics-related properties
    and methods.

    """

    def __init__(self) -> None:
        """Initialization method."""

        self.algorithm = self.__class__.__name__
        self.params = {}

        self.built = False

    @property
    def algorithm(self) -> str:
        """str: Algorithm's name."""

        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: str) -> None:
        if not isinstance(algorithm, str):
            raise e.TypeError("`algorithm` should be a string")

        self._algorithm = algorithm

    @property
    def built(self) -> bool:
        """Indicates whether the optimizer is built."""

        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        if not isinstance(built, bool):
            raise e.TypeError("`built` should be a boolean")

        self._built = built

    @property
    def params(self) -> Dict[str, Any]:
        """Key-value parameters."""

        return self._params

    @params.setter
    def params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise e.TypeError("`params` should be a dictionary")

        self._params = params

    def build(self, params: Dict[str, Any]) -> None:
        """Builds the object by creating its parameters.

        Args:
            params: Key-value parameters to the meta-heuristic.

        """

        if params:
            self.params = params

            for k, v in params.items():
                setattr(self, k, v)

        self.built = True

        logger.debug(
            "Algorithm: %s | Custom Parameters: %s | Built: %s.",
            self.algorithm,
            str(params),
            self.built,
        )

    def compile(self, space: Union[_SingleObjectiveSpace, _SingleObjectiveTensorSpace,
                                   _MultiObjectiveSpace, _MultiObjectiveTensorSpace]) -> None:
        """Compiles additional information that is used by this optimizer.

        This method is called before the optimization procedure and makes sure
        that the additional variable is available as a property.

        """

        pass

    def evaluate(self, space: Union[_SingleObjectiveSpace, _SingleObjectiveTensorSpace], function: Function) -> None:
        """Evaluates the search space according to the objective function.

        If you need a specific evaluate method, please re-implement
        it on child's class.

        Also, note that function only accept arguments that are
        found on Opytimizer class.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object serving as an objective function.

        """

        for agent in space.agents:
            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Union[_SingleObjectiveSpace, _SingleObjectiveTensorSpace, _MultiObjectiveSpace,
                                  _MultiObjectiveTensorSpace], function: Function) -> None:
        """Updates the agents' position array.

        As each child has a different procedure of update, you will need
        to implement it directly on its class.


        """

        pass




class MultiObjectiveOptimizer(Optimizer):
    """A MultiObjectiveOptimizer class that holds multi-objective meta-heuristics-related
    properties and methods.

    """

    def __init__(self) -> None:
        """Initialization method."""

        super().__init__()

    def evaluate(self, space:Union[_MultiObjectiveSpace, _MultiObjectiveTensorSpace], function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object serving as an objective function.

        """

        for agent in space.agents:
            agent.fit = function(agent.position)

        

class TensorizedOptimizer:
    def sync(self, space: _SingleObjectiveTensorSpace) -> None:
            """
                Converts tensorized population to Opytimizer's default structure.
            """
            xp = space.env.xp

            if xp.__name__ == 'cupy':
                for i, agent in enumerate(space.agents):
                               
                    agent.position[:] = space.X[i].reshape(-1, 1).get()
                                
                    agent.fit = float(space.F[i]) 
        

                space.best_agent.position = self.global_best_position.reshape(-1, 1).get()
                space.best_agent.fit = space.best_agent.fit.get()

            else:
                for i, agent in enumerate(space.agents):
                                               
                    agent.position[:] = space.X[i].reshape(-1, 1)
                                                
                    agent.fit = float(space.F[i]) 
                        
                
                space.best_agent.position[:] = self.global_best_position.reshape(-1, 1)

             
    
                

class TensorizedMultiObjectiveOptimizer:
    def sync(self, space: _MultiObjectiveTensorSpace):
            """
                Converts tensorized population to Opytimizer's default structure.
            """ 
            xp = space.env.xp
            if xp.__name__ == 'cupy':
                for i, agent in enumerate(space.agents):
                    agent.position[:] = xp.array(space.X[i]).reshape(agent.position.shape).get()
                    agent.fit[:] = space.F[i].get()
            else:
                for i, agent in enumerate(space.agents):
                    agent.position[:] = xp.array(space.X[i]).reshape(agent.position.shape)
                    agent.fit[:] = space.F[i]
        

