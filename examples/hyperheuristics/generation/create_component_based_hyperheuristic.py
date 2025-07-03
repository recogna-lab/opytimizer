import numpy as np

from opytimizer.hyperheuristics.generation import ComponentBasedHyperHeuristic
from opytimizer.optimizers.single_objective.swarm import ABC, PSO

# One should declare a list of optimizers to be used by the hyperheuristic
optimizers = [PSO, ABC]

# One should declare hyperparameters for each optimizer
optimizer_params = [
    {"w": 0.7, "c1": 1.7, "c2": 1.7},  # PSO
    {"n_trials": 10},  # ABC
]

# One should declare available components
def custom_selection(agents):
    """Custom selection component."""
    return sorted(agents, key=lambda x: x.fit)[: len(agents) // 2]


def custom_crossover(parent1, parent2):
    """Custom crossover component."""
    alpha = 0.5
    child1 = alpha * parent1.position + (1 - alpha) * parent2.position
    child2 = (1 - alpha) * parent1.position + alpha * parent2.position
    return child1, child2


def custom_mutation(agent):
    """Custom mutation component."""
    noise = np.random.normal(0, 0.1, agent.position.shape)
    agent.position += noise


components = {
    "selection": [custom_selection],
    "crossover": [custom_crossover],
    "mutation": [custom_mutation],
}

# One should declare hyperheuristic parameters
params = {
    "components": components,
    "population_size": 40,
    "max_components": 4,
    "mutation_rate": 0.15,
    "crossover_rate": 0.7,
    "max_generations": 80,
}

# Creates a ComponentBasedHyperHeuristic
h = ComponentBasedHyperHeuristic(
    components=components, population_size=40, crossover_rate=0.7, mutation_rate=0.15
)
