"""Component-Based Hyperheuristic implementation.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.hyperheuristic import HyperHeuristic
from opytimizer.core.space import _Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class ComponentBasedHyperHeuristic(HyperHeuristic):
    """A Component-Based Hyperheuristic that combines different
    components from various optimization algorithms.

    This hyperheuristic generates new algorithms by selecting and
    combining different components (initialization, selection,
    variation, etc.) from existing algorithms.
    """

    def __init__(
        self,
        components: Optional[Dict[str, List[Callable]]] = None,
        population_size: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        performance_metric=None,
    ) -> None:
        """Initialization method.

        Args:
            components: Dictionary of algorithm components.
            population_size: Size of the population.
            crossover_rate: Probability of crossover.
            mutation_rate: Probability of mutation.
            performance_metric: Function to evaluate performance.
        """
        super().__init__(performance_metric=performance_metric)

        self.components = components or self._default_components()
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.population = []
        self.fitness_history = []
        self.best_algorithm = None
        self.best_fitness = float("inf")

    def _default_components(self) -> Dict[str, List[Callable]]:
        """Default components for algorithm generation."""
        return {
            "initialization": [
                self._random_initialization,
                self._uniform_initialization,
                self._gaussian_initialization,
            ],
            "selection": [
                self._tournament_selection,
                self._roulette_selection,
                self._rank_selection,
            ],
            "crossover": [
                self._uniform_crossover,
                self._single_point_crossover,
                self._arithmetic_crossover,
            ],
            "mutation": [
                self._gaussian_mutation,
                self._uniform_mutation,
                self._swap_mutation,
            ],
            "replacement": [
                self._generational_replacement,
                self._elitist_replacement,
                self._steady_state_replacement,
            ],
        }

    def _random_initialization(self, space: _Space) -> None:
        """Random initialization component."""
        for agent in space.agents:
            agent.position = np.random.uniform(
                space.lb, space.ub, size=agent.position.shape
            )

    def _uniform_initialization(self, space: _Space) -> None:
        """Uniform initialization component."""
        for agent in space.agents:
            agent.position = np.random.uniform(
                space.lb, space.ub, size=agent.position.shape
            )

    def _gaussian_initialization(self, space: _Space) -> None:
        """Gaussian initialization component."""
        for agent in space.agents:
            mean = (space.lb + space.ub) / 2
            std = (space.ub - space.lb) / 4
            agent.position = np.random.normal(mean, std, size=agent.position.shape)
            agent.position = np.clip(agent.position, space.lb, space.ub)

    def _tournament_selection(
        self, agents: List[Any], tournament_size: int = 3
    ) -> List[Any]:
        """Tournament selection component."""
        selected = []
        for _ in range(len(agents)):
            tournament = random.sample(agents, tournament_size)
            winner = min(tournament, key=lambda x: x.fit)
            selected.append(winner)
        return selected

    def _roulette_selection(self, agents: List[Any]) -> List[Any]:
        """Roulette wheel selection component."""
        fitness_values = [agent.fit for agent in agents]
        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            return random.sample(agents, len(agents))

        probabilities = [f / total_fitness for f in fitness_values]
        selected = []
        for _ in range(len(agents)):
            selected.append(random.choices(agents, weights=probabilities)[0])
        return selected

    def _rank_selection(self, agents: List[Any]) -> List[Any]:
        """Rank-based selection component."""
        sorted_agents = sorted(agents, key=lambda x: x.fit)
        ranks = list(range(1, len(agents) + 1))
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]

        selected = []
        for _ in range(len(agents)):
            selected.append(random.choices(sorted_agents, weights=probabilities)[0])
        return selected

    def _uniform_crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Uniform crossover component."""
        mask = np.random.random(parent1.position.shape) < 0.5
        child1 = parent1.position.copy()
        child2 = parent2.position.copy()

        child1[mask] = parent2.position[mask]
        child2[mask] = parent1.position[mask]

        return child1, child2

    def _single_point_crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Single-point crossover component."""
        point = random.randint(1, len(parent1.position) - 1)
        child1 = np.concatenate([parent1.position[:point], parent2.position[point:]])
        child2 = np.concatenate([parent2.position[:point], parent1.position[point:]])

        return child1, child2

    def _arithmetic_crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Arithmetic crossover component."""
        alpha = random.random()
        child1 = alpha * parent1.position + (1 - alpha) * parent2.position
        child2 = (1 - alpha) * parent1.position + alpha * parent2.position

        return child1, child2

    def _gaussian_mutation(self, agent: Any, mutation_rate: float = 0.1) -> None:
        """Gaussian mutation component."""
        mask = np.random.random(agent.position.shape) < mutation_rate
        noise = np.random.normal(0, 0.1, agent.position.shape)
        agent.position[mask] += noise[mask]

    def _uniform_mutation(self, agent: Any, mutation_rate: float = 0.1) -> None:
        """Uniform mutation component."""
        mask = np.random.random(agent.position.shape) < mutation_rate
        agent.position[mask] = np.random.uniform(-1, 1, size=agent.position[mask].shape)

    def _swap_mutation(self, agent: Any, mutation_rate: float = 0.1) -> None:
        """Swap mutation component."""
        if random.random() < mutation_rate:
            indices = random.sample(range(len(agent.position)), 2)
            agent.position[indices[0]], agent.position[indices[1]] = (
                agent.position[indices[1]],
                agent.position[indices[0]],
            )

    def _generational_replacement(
        self, parents: List[Any], offspring: List[Any]
    ) -> List[Any]:
        """Generational replacement component."""
        return offspring

    def _elitist_replacement(
        self, parents: List[Any], offspring: List[Any]
    ) -> List[Any]:
        """Elitist replacement component."""
        best_parent = min(parents, key=lambda x: x.fit)
        offspring[0] = best_parent
        return offspring

    def _steady_state_replacement(
        self, parents: List[Any], offspring: List[Any]
    ) -> List[Any]:
        """Steady-state replacement component."""
        all_individuals = parents + offspring
        return sorted(all_individuals, key=lambda x: x.fit)[: len(parents)]

    def _create_random_algorithm(self) -> Dict[str, Any]:
        """Create a random algorithm by combining components.

        Returns:
            (Dict[str, Any]): Algorithm configuration.
        """
        algorithm = {}
        for component_type, component_list in self.components.items():
            algorithm[component_type] = random.choice(component_list)
        return algorithm

    def _evaluate_algorithm(self, algorithm: Dict[str, Any], space: _Space) -> float:
        """Evaluate a generated algorithm.

        Args:
            algorithm: Algorithm configuration.
            space: Search space.

        Returns:
            (float): Fitness of the algorithm.
        """
        try:
            # Execute the algorithm for a few iterations
            original_agents = [agent.copy() for agent in space.agents]

            # Initialize
            if "initialization" in algorithm:
                algorithm["initialization"](space)

            # Run a few iterations
            for _ in range(5):
                # Selection
                if "selection" in algorithm:
                    algorithm["selection"](space.agents)

                # Crossover
                if "crossover" in algorithm and len(space.agents) >= 2:
                    for i in range(0, len(space.agents) - 1, 2):
                        child1, child2 = algorithm["crossover"](
                            space.agents[i], space.agents[i + 1]
                        )
                        space.agents[i].position = child1
                        space.agents[i + 1].position = child2

                # Mutation
                if "mutation" in algorithm:
                    for agent in space.agents:
                        algorithm["mutation"](agent)

            # Calculate improvement
            original_fitness = min([agent.fit for agent in original_agents])
            final_fitness = min([agent.fit for agent in space.agents])
            improvement = original_fitness - final_fitness

            # Restore original agents
            space.agents = original_agents

            return -improvement  # Minimization problem
        except Exception:
            return float("inf")

    def _initialize_population(self) -> None:
        """Initialize the population with random algorithms."""
        self.population = []
        for _ in range(self.population_size):
            algorithm = self._create_random_algorithm()
            self.population.append(
                {"algorithm": algorithm, "fitness": float("inf"), "age": 0}
            )

    def _select_tournament(self) -> Dict[str, Any]:
        """Perform tournament selection.

        Returns:
            (Dict[str, Any]): Selected individual.
        """
        tournament = random.sample(self.population, 3)
        return min(tournament, key=lambda x: x["fitness"])

    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            (Tuple[Dict[str, Any], Dict[str, Any]]): Two offspring.
        """
        if random.random() < self.crossover_rate:
            # Uniform crossover for algorithm components
            offspring1 = {}
            offspring2 = {}

            for component_type in self.components.keys():
                if random.random() < 0.5:
                    offspring1[component_type] = parent1["algorithm"][component_type]
                    offspring2[component_type] = parent2["algorithm"][component_type]
                else:
                    offspring1[component_type] = parent2["algorithm"][component_type]
                    offspring2[component_type] = parent1["algorithm"][component_type]

            return (
                {"algorithm": offspring1, "fitness": float("inf"), "age": 0},
                {"algorithm": offspring2, "fitness": float("inf"), "age": 0},
            )
        else:
            return parent1.copy(), parent2.copy()

    def _mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mutation on an individual.

        Args:
            individual: Individual to mutate.

        Returns:
            (Dict[str, Any]): Mutated individual.
        """
        if random.random() < self.mutation_rate:
            algorithm = individual["algorithm"].copy()

            # Mutate a random component
            component_type = random.choice(list(self.components.keys()))
            algorithm[component_type] = random.choice(self.components[component_type])

            return {
                "algorithm": algorithm,
                "fitness": float("inf"),
                "age": individual["age"] + 1,
            }
        return individual

    def compile(self, space: _Space) -> None:
        """Compile the component-based hyperheuristic.

        Args:
            space: A Space object containing meta-information.
        """
        self._initialize_population()
        logger.debug(
            "Initialized component-based population with %d individuals.",
            self.population_size,
        )

    def evaluate(self, space: _Space, function: Function) -> None:
        """Evaluate the population of generated algorithms.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object serving as an objective function.
        """
        # Evaluate all individuals in the population
        for individual in self.population:
            individual["fitness"] = self._evaluate_algorithm(
                individual["algorithm"], space
            )

        # Update best individual
        best_individual = min(self.population, key=lambda x: x["fitness"])
        if best_individual["fitness"] < self.best_fitness:
            self.best_fitness = best_individual["fitness"]
            self.best_algorithm = best_individual.copy()

        # Record fitness history
        avg_fitness = np.mean([ind["fitness"] for ind in self.population])
        self.fitness_history.append(avg_fitness)

    def update(self, space: _Space) -> None:
        """Update the population through evolution.

        Args:
            space: A Space object containing agents and update-related information.
        """
        # Create new population through selection, crossover, and mutation
        new_population = []

        # Elitism: keep best individual
        best_individual = min(self.population, key=lambda x: x["fitness"])
        new_population.append(best_individual.copy())

        # Generate rest of population
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._select_tournament()
                parent2 = self._select_tournament()
                offspring1, offspring2 = self._crossover(parent1, parent2)
                new_population.extend([offspring1, offspring2])
            else:
                # Mutation
                parent = self._select_tournament()
                offspring = self._mutation(parent)
                new_population.append(offspring)

        # Trim population to size
        self.population = new_population[: self.population_size]

        # Increment iteration
        self.iteration += 1

    def get_component_statistics(self) -> Dict[str, Any]:
        """Get statistics about the component-based evolution.

        Returns:
            (Dict[str, Any]): Dictionary containing component statistics.
        """
        stats = super().get_statistics()
        stats.update(
            {
                "population_size": self.population_size,
                "best_fitness": self.best_fitness,
                "average_fitness": np.mean([ind["fitness"] for ind in self.population]),
                "fitness_history": self.fitness_history,
                "best_algorithm": self.best_algorithm,
                "component_types": list(self.components.keys()),
            }
        )
        return stats
