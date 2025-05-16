"""Chernobyl Disaster Optimizer.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.core import Optimizer
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class CDO(Optimizer):
    """An CDO class, inherited from Optimizer.

    This is the designed class to define CDO-related
    variables and methods.

    References:
        H. Abedinpourshotorban et al.
        Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
        Swarm and Evolutionary Computation (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(CDO, self).__init__()


        self.build(params)

        logger.info("Class overrided.")

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.gamma_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.gamma_fit = c.FLOAT_MAX

        self.beta_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.beta_fit = c.FLOAT_MAX

        self.alpha_pos = np.zeros((space.n_variables, space.n_dimensions))
        self.alpha_fit = c.FLOAT_MAX

    def update(self, space: Space, function: Function, iteration: int, n_iterations: int) -> None:
        """Wraps Chernobyl Disaster Optimizer over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        for agent in space.agents:

            fit = function(agent.position)

            if fit < self.alpha_fit:
                self.alpha_fit = fit
                self.alpha_pos = copy.deepcopy(agent.position)

            if fit < self.alpha_fit and fit < self.beta_fit:
                self.beta_fit = fit
                self.beta_pos = copy.deepcopy(agent.position)

            if fit < self.alpha_fit and fit < self.beta_fit and fit < self.gamma_fit:
                self.gamma_fit = fit
                self.gamma_pos = copy.deepcopy(agent.position)

        ws = 3 - 3 * iteration/n_iterations
        s_gamma = np.log10(r.generate_uniform_random_number(1, 300000))
        s_beta = np.log10(r.generate_uniform_random_number(1, 270000))
        s_alpha = np.log10(r.generate_uniform_random_number(1, 16000))

        for agent in space.agents:

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_gamma = np.pi * r1 * r1 / s_gamma - ws * r2
            a_gamma = r3 * r3 * np.pi
            grad_gamma = np.abs(a_gamma * self.gamma_pos - agent.position)
            v_gamma = agent.position - rho_gamma * grad_gamma

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_beta = np.pi * r1 * r1 / (0.5 * s_beta) - ws * r2
            a_beta = r3 * r3 * np.pi
            grad_beta = np.abs(a_beta * self.beta_pos - agent.position)
            v_beta = 0.5 * (agent.position - rho_beta * grad_beta)

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_alpha = np.pi * r1 * r1 / (0.25 * s_alpha) - ws * r2
            a_alpha = r3 * r3 * np.pi
            grad_alpha = np.abs(a_alpha * self.alpha_pos - agent.position)
            v_alpha = 0.25 * (agent.position - rho_alpha * grad_alpha)

            agent.position = (v_alpha + v_beta + v_gamma) / 3


class OBCDO(CDO):
    """Opposition-Based Chernobyl Disaster Optimizer.
    
    This variant implements multiple Opposition-Based Learning strategies:
    - Basic OBL (BOBL)
    - Quasi OBL (QOBL)
    - Generalized OBL (GOBL)
    - Partial OBL (POBL)
    - Center-Based OBL (COBL)
    - Enhanced OBL (EOBL)
    - Time-Varying OBL (TVOBL)
    - Elite OBL (Elite-OBL)
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
        """
        super(OBCDO, self).__init__(params)
        
        # Opposition parameters
        self.obl_rate = params.get("obl_rate", 0.3)
        self.min_obl_rate = params.get("min_obl_rate", 0.1)
        self.max_obl_rate = params.get("max_obl_rate", 0.5)
        self.use_quasi = params.get("use_quasi", True)
        self.success_memory = []
        self.memory_size = params.get("memory_size", 10)
        
        # Strategy parameters
        self.obl_strategy = params.get("obl_strategy", "adaptive")
        self.partial_dims = params.get("partial_dims", 0.5)  # Percentage of dimensions
        self.generalized_factor = params.get("generalized_factor", 0.5)
        self.elite_size = params.get("elite_size", 3)
        self.elite_solutions = []
        self.center_position = None
        
        # Strategy success tracking
        self.strategy_success = {
            "basic": 0,
            "quasi": 0,
            "generalized": 0,
            "partial": 0,
            "center": 0,
            "enhanced": 0,
            "time_varying": 0,
            "elite": 0
        }
        
    def get_opposite_position(self, position: np.ndarray, space: Space) -> np.ndarray:
        """Basic Opposition-Based Learning (BOBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            
        Returns:
            Opposite position
        """
        return space.lb + space.ub - position
        
    def get_quasi_opposite_position(self, position: np.ndarray, space: Space) -> np.ndarray:
        """Quasi Opposition-Based Learning (QOBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            
        Returns:
            Quasi-opposite position
        """
        opposite = self.get_opposite_position(position, space)
        mean = (position + opposite) / 2
        rand = r.generate_uniform_random_number(
            size=(space.n_variables, space.n_dimensions)
        )
        return mean + rand * (opposite - mean)
        
    def get_generalized_opposite_position(self, position: np.ndarray, space: Space) -> np.ndarray:
        """Generalized Opposition-Based Learning (GOBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            
        Returns:
            Generalized opposite position
        """
        k = self.generalized_factor
        mean = (space.lb + space.ub) / 2
        return k * (space.lb + space.ub) - position
        
    def get_partial_opposite_position(self, position: np.ndarray, space: Space) -> np.ndarray:
        """Partial Opposition-Based Learning (POBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            
        Returns:
            Partial opposite position
        """
        opposite = position.copy()
        n_dims = position.shape[1]
        dims_to_oppose = int(n_dims * self.partial_dims)
        dims = np.random.choice(n_dims, dims_to_oppose, replace=False)
        
        for dim in dims:
            opposite[:, dim] = space.lb[dim] + space.ub[dim] - position[:, dim]
            
        return opposite
        
    def get_center_opposite_position(self, position: np.ndarray, space: Space) -> np.ndarray:
        """Center-Based Opposition-Based Learning (COBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            
        Returns:
            Center-based opposite position
        """
        if self.center_position is None:
            self.center_position = (space.lb + space.ub) / 2
            
        return self.center_position + (self.center_position - position)
        
    def get_enhanced_opposite_position(self, position: np.ndarray, space: Space) -> np.ndarray:
        """Enhanced Opposition-Based Learning (EOBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            
        Returns:
            Enhanced opposite position
        """
        # Combine multiple strategies
        strategies = [
            self.get_opposite_position,
            self.get_quasi_opposite_position,
            self.get_generalized_opposite_position,
            self.get_center_opposite_position
        ]
        strategy_names = ["basic", "quasi", "generalized", "center"]
        
        # Select strategy based on success rate
        total_success = sum(self.strategy_success[s] for s in strategy_names) + 1e-10
        probs = np.array([self.strategy_success[s] for s in strategy_names])
        probs = probs / total_success
        
        # If all probabilities are zero, use uniform distribution
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)
            
        strategy = np.random.choice(strategies, p=probs)
        
        return strategy(position, space)
        
    def get_time_varying_opposite_position(self, position: np.ndarray, space: Space, iteration: int, n_iterations: int) -> np.ndarray:
        """Time-Varying Opposition-Based Learning (TVOBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            iteration: Current iteration
            n_iterations: Maximum iterations
            
        Returns:
            Time-varying opposite position
        """
        progress = iteration / n_iterations
        if progress < 0.3:  # Early exploration
            return self.get_opposite_position(position, space)
        elif progress < 0.7:  # Mid-stage balance
            return self.get_quasi_opposite_position(position, space)
        else:  # Late exploitation
            return self.get_center_opposite_position(position, space)
            
    def get_elite_opposite_position(self, position: np.ndarray, space: Space) -> np.ndarray:
        """Elite Opposition-Based Learning (Elite-OBL).
        
        Args:
            position: Current position
            space: Space object containing bounds
            
        Returns:
            Elite-based opposite position
        """
        if not self.elite_solutions:
            return self.get_opposite_position(position, space)
            
        # Use elite solutions to guide opposition
        elite_center = np.mean([sol for sol in self.elite_solutions], axis=0)
        return elite_center + (elite_center - position)
        
    def update_elite_solutions(self, space: Space, function: Function) -> None:
        """Update elite solutions pool."""
        solutions = [(agent.position, function(agent.position)) for agent in space.agents]
        solutions.sort(key=lambda x: x[1])
        self.elite_solutions = [sol[0] for sol in solutions[:self.elite_size]]
        
    def select_strategy(self, iteration: int, n_iterations: int) -> str:
        """Select OBL strategy based on current state.
        
        Args:
            iteration: Current iteration
            n_iterations: Maximum iterations
            
        Returns:
            Selected strategy name
        """
        if self.obl_strategy == "adaptive":
            # Select based on success rates
            total_success = sum(self.strategy_success.values()) + 1e-10
            strategies = list(self.strategy_success.keys())
            probs = np.array([self.strategy_success[s] for s in strategies])
            probs = probs / total_success  # Normalize probabilities
            
            # If all probabilities are zero, use uniform distribution
            if np.all(probs == 0):
                probs = np.ones_like(probs) / len(probs)
                
            return np.random.choice(strategies, p=probs)
        elif self.obl_strategy == "time_varying":
            progress = iteration / n_iterations
            if progress < 0.3:
                return "basic"
            elif progress < 0.7:
                return "quasi"
            else:
                return "center"
        else:
            return self.obl_strategy
            
    def adapt_obl_rate(self, success_rate: float) -> None:
        """Adapt opposition rate based on success rate.
        
        Args:
            success_rate: Rate of successful oppositions
        """
        if success_rate > 0.6:  # Too successful, increase exploration
            self.obl_rate = min(self.max_obl_rate, self.obl_rate * 1.1)
        elif success_rate < 0.2:  # Too unsuccessful, decrease exploration
            self.obl_rate = max(self.min_obl_rate, self.obl_rate * 0.9)
            
    def update(self, space: Space, function: Function, iteration: int, n_iterations: int) -> None:
        """Updates using Opposition-Based Learning.
        
        Args:
            space: Space containing agents and update-related information
            function: Objective function
            iteration: Current iteration
            n_iterations: Maximum iterations
        """
        # Regular CDO update
        super().update(space, function, iteration, n_iterations)
        
        # Update elite solutions if using Elite-OBL
        if self.obl_strategy in ["elite", "adaptive"]:
            self.update_elite_solutions(space, function)
        
        success_count = 0
        total_count = 0
        
        # Apply opposition with probability
        if r.generate_uniform_random_number() < self.obl_rate:
            strategy = self.select_strategy(iteration, n_iterations)
            
            for agent in space.agents:
                total_count += 1
                current_fit = function(agent.position)
                
                # Generate opposite position based on selected strategy
                if strategy == "basic":
                    opposite_pos = self.get_opposite_position(agent.position, space)
                elif strategy == "quasi":
                    opposite_pos = self.get_quasi_opposite_position(agent.position, space)
                elif strategy == "generalized":
                    opposite_pos = self.get_generalized_opposite_position(agent.position, space)
                elif strategy == "partial":
                    opposite_pos = self.get_partial_opposite_position(agent.position, space)
                elif strategy == "center":
                    opposite_pos = self.get_center_opposite_position(agent.position, space)
                elif strategy == "enhanced":
                    opposite_pos = self.get_enhanced_opposite_position(agent.position, space)
                elif strategy == "time_varying":
                    opposite_pos = self.get_time_varying_opposite_position(agent.position, space, iteration, n_iterations)
                else:  # elite
                    opposite_pos = self.get_elite_opposite_position(agent.position, space)
                    
                opposite_fit = function(opposite_pos)
                
                if opposite_fit < current_fit:
                    agent.position = opposite_pos
                    success_count += 1
                    self.strategy_success[strategy] += 1
                    
        # Update success rate and adapt opposition rate
        if total_count > 0:
            success_rate = success_count / total_count
            self.success_memory.append(success_rate)
            if len(self.success_memory) > self.memory_size:
                self.success_memory.pop(0)
                
            if len(self.success_memory) >= self.memory_size:
                avg_success_rate = np.mean(self.success_memory)
                self.adapt_obl_rate(avg_success_rate)


class ChaoticCDO(CDO):
    """Chaotic Chernobyl Disaster Optimizer.
    
    This variant uses chaotic maps to improve the search behavior.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
        """
        super(ChaoticCDO, self).__init__(params)
        
        # Chaotic map initial value
        self.chaotic_value = params.get("chaotic_value", 0.7)
        self.map_type = params.get("map_type", "logistic")
        self.adaptive_maps = params.get("adaptive_maps", True)
        self.map_success = {"logistic": 0, "sine": 0, "tent": 0}
        self.current_map = self.map_type
        
    def logistic_map(self) -> float:
        """Implements logistic map for chaos generation."""
        self.chaotic_value = 4 * self.chaotic_value * (1 - self.chaotic_value)
        return self.chaotic_value
        
    def sine_map(self) -> float:
        """Implements sine map for chaos generation."""
        self.chaotic_value = 2.3 * np.sin(np.pi * self.chaotic_value)
        return self.chaotic_value
        
    def tent_map(self) -> float:
        """Implements tent map for chaos generation."""
        if self.chaotic_value < 0.5:
            self.chaotic_value = 2 * self.chaotic_value
        else:
            self.chaotic_value = 2 * (1 - self.chaotic_value)
        return self.chaotic_value
        
    def get_chaotic_value(self) -> float:
        """Get chaotic value based on current map type."""
        if self.current_map == "logistic":
            return self.logistic_map()
        elif self.current_map == "sine":
            return self.sine_map()
        else:  # tent
            return self.tent_map()
            
    def update_map_selection(self, success: bool) -> None:
        """Update map selection based on success."""
        if success:
            self.map_success[self.current_map] += 1
            
        if self.adaptive_maps and np.random.random() < 0.1:  # 10% chance to switch
            total_success = sum(self.map_success.values()) + 1e-10
            probs = [v/total_success for v in self.map_success.values()]
            self.current_map = np.random.choice(list(self.map_success.keys()), p=probs)
        
    def update(self, space: Space, function: Function, iteration: int, n_iterations: int) -> None:
        """Updates using chaotic values.
        
        Args:
            space: Space containing agents and update-related information
            function: Objective function
            iteration: Current iteration
            n_iterations: Maximum iterations
        """
        improved = False
        
        for agent in space.agents:
            old_fit = function(agent.position)
            fit = function(agent.position)

            if fit < self.alpha_fit:
                self.alpha_fit = fit
                self.alpha_pos = copy.deepcopy(agent.position)
                improved = True

            if fit < self.alpha_fit and fit < self.beta_fit:
                self.beta_fit = fit
                self.beta_pos = copy.deepcopy(agent.position)

            if fit < self.alpha_fit and fit < self.beta_fit and fit < self.gamma_fit:
                self.gamma_fit = fit
                self.gamma_pos = copy.deepcopy(agent.position)

        ws = 3 - 3 * iteration/n_iterations
        
        # Use chaotic values instead of random
        s_gamma = np.log10(300000 * self.get_chaotic_value())
        s_beta = np.log10(270000 * self.get_chaotic_value())
        s_alpha = np.log10(16000 * self.get_chaotic_value())

        for agent in space.agents:
            # Use chaotic values for r1, r2, r3
            r1 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))
            r2 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))
            r3 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))

            rho_gamma = np.pi * r1 * r1 / s_gamma - ws * r2
            a_gamma = r3 * r3 * np.pi
            grad_gamma = np.abs(a_gamma * self.gamma_pos - agent.position)
            v_gamma = agent.position - rho_gamma * grad_gamma

            r1 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))
            r2 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))
            r3 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))

            rho_beta = np.pi * r1 * r1 / (0.5 * s_beta) - ws * r2
            a_beta = r3 * r3 * np.pi
            grad_beta = np.abs(a_beta * self.beta_pos - agent.position)
            v_beta = 0.5 * (agent.position - rho_beta * grad_beta)

            r1 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))
            r2 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))
            r3 = self.get_chaotic_value() * np.ones((space.n_variables, space.n_dimensions))

            rho_alpha = np.pi * r1 * r1 / (0.25 * s_alpha) - ws * r2
            a_alpha = r3 * r3 * np.pi
            grad_alpha = np.abs(a_alpha * self.alpha_pos - agent.position)
            v_alpha = 0.25 * (agent.position - rho_alpha * grad_alpha)

            old_pos = copy.deepcopy(agent.position)
            agent.position = (v_alpha + v_beta + v_gamma) / 3
            
            if function(agent.position) < function(old_pos):
                improved = True
                
        # Update map selection based on improvement
        self.update_map_selection(improved)


class MultiReactorCDO(CDO):
    """Multi-Reactor Chernobyl Disaster Optimizer.
    
    This variant uses multiple sub-populations (reactors) that occasionally exchange information.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
        """
        super(MultiReactorCDO, self).__init__(params)
        
        # Number of reactors
        self.n_reactors = params.get("n_reactors", 3)
        # Exchange interval
        self.exchange_interval = params.get("exchange_interval", 10)
        
        # Initialize reactor-specific variables
        self.reactor_gamma_pos = []
        self.reactor_gamma_fit = []
        self.reactor_beta_pos = []
        self.reactor_beta_fit = []
        self.reactor_alpha_pos = []
        self.reactor_alpha_fit = []
        
    def compile(self, space: Space) -> None:
        """Compiles additional information for each reactor.

        Args:
            space: Space object containing meta-information.
        """
        for _ in range(self.n_reactors):
            self.reactor_gamma_pos.append(np.zeros((space.n_variables, space.n_dimensions)))
            self.reactor_gamma_fit.append(c.FLOAT_MAX)
            self.reactor_beta_pos.append(np.zeros((space.n_variables, space.n_dimensions)))
            self.reactor_beta_fit.append(c.FLOAT_MAX)
            self.reactor_alpha_pos.append(np.zeros((space.n_variables, space.n_dimensions)))
            self.reactor_alpha_fit.append(c.FLOAT_MAX)
            
    def exchange_information(self) -> None:
        """Exchange best solutions between reactors."""
        # Find best reactor
        best_reactor = np.argmin([self.reactor_alpha_fit[i] for i in range(self.n_reactors)])
        
        # Share best solution with other reactors
        for i in range(self.n_reactors):
            if i != best_reactor:
                if self.reactor_alpha_fit[best_reactor] < self.reactor_gamma_fit[i]:
                    self.reactor_gamma_pos[i] = copy.deepcopy(self.reactor_alpha_pos[best_reactor])
                    self.reactor_gamma_fit[i] = self.reactor_alpha_fit[best_reactor]
        
    def update(self, space: Space, function: Function, iteration: int, n_iterations: int) -> None:
        """Updates using multiple reactors.
        
        Args:
            space: Space containing agents and update-related information
            function: Objective function
            iteration: Current iteration
            n_iterations: Maximum iterations
        """
        # Split population into reactors
        agents_per_reactor = len(space.agents) // self.n_reactors
        
        for reactor in range(self.n_reactors):
            start_idx = reactor * agents_per_reactor
            end_idx = start_idx + agents_per_reactor
            
            # Update each reactor independently
            for agent in space.agents[start_idx:end_idx]:
                fit = function(agent.position)

                if fit < self.reactor_alpha_fit[reactor]:
                    self.reactor_alpha_fit[reactor] = fit
                    self.reactor_alpha_pos[reactor] = copy.deepcopy(agent.position)

                if fit < self.reactor_alpha_fit[reactor] and fit < self.reactor_beta_fit[reactor]:
                    self.reactor_beta_fit[reactor] = fit
                    self.reactor_beta_pos[reactor] = copy.deepcopy(agent.position)

                if (fit < self.reactor_alpha_fit[reactor] and 
                    fit < self.reactor_beta_fit[reactor] and 
                    fit < self.reactor_gamma_fit[reactor]):
                    self.reactor_gamma_fit[reactor] = fit
                    self.reactor_gamma_pos[reactor] = copy.deepcopy(agent.position)

            ws = 3 - 3 * iteration/n_iterations
            s_gamma = np.log10(r.generate_uniform_random_number(1, 300000))
            s_beta = np.log10(r.generate_uniform_random_number(1, 270000))
            s_alpha = np.log10(r.generate_uniform_random_number(1, 16000))

            for agent in space.agents[start_idx:end_idx]:
                r1 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )
                r2 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )
                r3 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )

                rho_gamma = np.pi * r1 * r1 / s_gamma - ws * r2
                a_gamma = r3 * r3 * np.pi
                grad_gamma = np.abs(a_gamma * self.reactor_gamma_pos[reactor] - agent.position)
                v_gamma = agent.position - rho_gamma * grad_gamma

                r1 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )
                r2 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )
                r3 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )

                rho_beta = np.pi * r1 * r1 / (0.5 * s_beta) - ws * r2
                a_beta = r3 * r3 * np.pi
                grad_beta = np.abs(a_beta * self.reactor_beta_pos[reactor] - agent.position)
                v_beta = 0.5 * (agent.position - rho_beta * grad_beta)

                r1 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )
                r2 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )
                r3 = r.generate_uniform_random_number(
                    size=(space.n_variables, space.n_dimensions)
                )

                rho_alpha = np.pi * r1 * r1 / (0.25 * s_alpha) - ws * r2
                a_alpha = r3 * r3 * np.pi
                grad_alpha = np.abs(a_alpha * self.reactor_alpha_pos[reactor] - agent.position)
                v_alpha = 0.25 * (agent.position - rho_alpha * grad_alpha)

                agent.position = (v_alpha + v_beta + v_gamma) / 3
                
        # Exchange information between reactors periodically
        if iteration % self.exchange_interval == 0:
            self.exchange_information()


class AdaptiveCDO(CDO):
    """Adaptive Chernobyl Disaster Optimizer.
    
    This variant implements adaptive parameter control mechanisms.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
        """
        super(AdaptiveCDO, self).__init__(params)
        
        # Adaptation parameters
        self.success_history = []
        self.history_size = params.get("history_size", 10)
        self.adaptation_rate = params.get("adaptation_rate", 0.1)
        
        # Parameter bounds
        self.s_gamma_min = np.log10(1000)
        self.s_gamma_max = np.log10(300000)
        self.s_beta_min = np.log10(1000)
        self.s_beta_max = np.log10(270000)
        self.s_alpha_min = np.log10(1000)
        self.s_alpha_max = np.log10(16000)
        
    def adapt_parameters(self, success_rate: float) -> None:
        """Adapt parameters based on success rate.
        
        Args:
            success_rate: Rate of successful updates
        """
        if success_rate > 0.2:  # Too successful, increase difficulty
            self.s_gamma_max *= (1 + self.adaptation_rate)
            self.s_beta_max *= (1 + self.adaptation_rate)
            self.s_alpha_max *= (1 + self.adaptation_rate)
        elif success_rate < 0.1:  # Too difficult, decrease
            self.s_gamma_min *= (1 - self.adaptation_rate)
            self.s_beta_min *= (1 - self.adaptation_rate)
            self.s_alpha_min *= (1 - self.adaptation_rate)
            
    def update(self, space: Space, function: Function, iteration: int, n_iterations: int) -> None:
        """Updates using adaptive parameters.
        
        Args:
            space: Space containing agents and update-related information
            function: Objective function
            iteration: Current iteration
            n_iterations: Maximum iterations
        """
        success_count = 0
        
        for agent in space.agents:
            old_fit = function(agent.position)
            old_pos = copy.deepcopy(agent.position)

            fit = function(agent.position)

            if fit < self.alpha_fit:
                self.alpha_fit = fit
                self.alpha_pos = copy.deepcopy(agent.position)
                success_count += 1

            if fit < self.alpha_fit and fit < self.beta_fit:
                self.beta_fit = fit
                self.beta_pos = copy.deepcopy(agent.position)

            if fit < self.alpha_fit and fit < self.beta_fit and fit < self.gamma_fit:
                self.gamma_fit = fit
                self.gamma_pos = copy.deepcopy(agent.position)

        ws = 3 - 3 * iteration/n_iterations
        
        # Use adaptive ranges for s parameters
        s_gamma = r.generate_uniform_random_number(self.s_gamma_min, self.s_gamma_max)
        s_beta = r.generate_uniform_random_number(self.s_beta_min, self.s_beta_max)
        s_alpha = r.generate_uniform_random_number(self.s_alpha_min, self.s_alpha_max)

        for agent in space.agents:
            old_pos = copy.deepcopy(agent.position)
            old_fit = function(old_pos)

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_gamma = np.pi * r1 * r1 / s_gamma - ws * r2
            a_gamma = r3 * r3 * np.pi
            grad_gamma = np.abs(a_gamma * self.gamma_pos - agent.position)
            v_gamma = agent.position - rho_gamma * grad_gamma

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_beta = np.pi * r1 * r1 / (0.5 * s_beta) - ws * r2
            a_beta = r3 * r3 * np.pi
            grad_beta = np.abs(a_beta * self.beta_pos - agent.position)
            v_beta = 0.5 * (agent.position - rho_beta * grad_beta)

            r1 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r2 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )
            r3 = r.generate_uniform_random_number(
                size=(space.n_variables, space.n_dimensions)
            )

            rho_alpha = np.pi * r1 * r1 / (0.25 * s_alpha) - ws * r2
            a_alpha = r3 * r3 * np.pi
            grad_alpha = np.abs(a_alpha * self.alpha_pos - agent.position)
            v_alpha = 0.25 * (agent.position - rho_alpha * grad_alpha)

            agent.position = (v_alpha + v_beta + v_gamma) / 3
            
            # Check if update was successful
            if function(agent.position) < old_fit:
                success_count += 1
                
        # Update success history
        success_rate = success_count / (len(space.agents) * 2)  # 2 updates per agent
        self.success_history.append(success_rate)
        if len(self.success_history) > self.history_size:
            self.success_history.pop(0)
            
        # Adapt parameters based on average success rate
        if iteration % self.history_size == 0:
            avg_success_rate = np.mean(self.success_history)
            self.adapt_parameters(avg_success_rate)


class QuantumCDO(CDO):
    """Quantum Chernobyl Disaster Optimizer.
    
    This variant implements quantum-inspired mechanisms for better exploration.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.
        """
        super(QuantumCDO, self).__init__(params)
        
        # Quantum parameters
        self.quantum_radius = params.get("quantum_radius", 0.1)
        self.superposition_rate = params.get("superposition_rate", 0.3)
        
    def quantum_position(self, center: np.ndarray, space: Space) -> np.ndarray:
        """Generate quantum position around center.
        
        Args:
            center: Center position for quantum cloud
            space: Space object containing bounds
            
        Returns:
            Quantum position
        """
        # Generate random angles
        theta = 2 * np.pi * r.generate_uniform_random_number(
            size=(space.n_variables, space.n_dimensions)
        )
        phi = np.pi * r.generate_uniform_random_number(
            size=(space.n_variables, space.n_dimensions)
        )
        
        # Calculate quantum displacement
        radius = self.quantum_radius * r.generate_uniform_random_number(
            size=(space.n_variables, space.n_dimensions)
        )
        
        # Convert to Cartesian coordinates
        displacement = radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Apply quantum displacement
        quantum_pos = center + displacement[0]  # Using only x-component for simplicity
        
        # Ensure bounds
        quantum_pos = np.clip(quantum_pos, space.lb, space.ub)
        
        return quantum_pos
        
    def update(self, space: Space, function: Function, iteration: int, n_iterations: int) -> None:
        """Updates using quantum-inspired mechanisms.
        
        Args:
            space: Space containing agents and update-related information
            function: Objective function
            iteration: Current iteration
            n_iterations: Maximum iterations
        """
        # Regular CDO update
        super().update(space, function, iteration, n_iterations)
        
        # Quantum update
        for agent in space.agents:
            if r.generate_uniform_random_number() < self.superposition_rate:
                # Create quantum superposition
                quantum_pos = self.quantum_position(agent.position, space)
                quantum_fit = function(quantum_pos)
                
                # Collapse to better state
                if quantum_fit < function(agent.position):
                    agent.position = quantum_pos
                    
                    # Update global best positions if needed
                    if quantum_fit < self.alpha_fit:
                        self.alpha_fit = quantum_fit
                        self.alpha_pos = copy.deepcopy(quantum_pos)
                        
                        if quantum_fit < self.beta_fit:
                            self.beta_fit = quantum_fit
                            self.beta_pos = copy.deepcopy(quantum_pos)
                            
                            if quantum_fit < self.gamma_fit:
                                self.gamma_fit = quantum_fit
                                self.gamma_pos = copy.deepcopy(quantum_pos)
