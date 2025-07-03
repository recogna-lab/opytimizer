from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.hyperheuristics.selection import SelectionHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import (
    ChoiceFunction,
    GreedySelection,
    MultiArmedBandit,
)
from opytimizer.optimizers.single_objective.swarm import ABC, PSO
from opytimizer.spaces import SearchSpace


# Define a performance metric function
def performance_metric(space):
    """Calculate performance based on best fitness value."""
    return space.best_agent.fit


# One should declare a list of optimizer instances to be used by the hyperheuristic
optimizers = [
    PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}),
    ABC(params={"n_trials": 10}),
]

# One should declare different selection strategies to compare
strategies = {
    "Choice Function": ChoiceFunction(),
    "Multi-Armed Bandit": MultiArmedBandit(),
    "Greedy": GreedySelection(),
}

# Creates a Function object
function = Function(Sphere())

# Creates a SearchSpace object
space = SearchSpace(
    n_agents=20,
    n_variables=2,
    n_objectives=1,
    lower_bound=[-10, -10],
    upper_bound=[10, 10],
)

# Test different selection strategies
for strategy_name, strategy in strategies.items():
    print(f"\n=== Testing {strategy_name} ===")

    # Creates a SelectionHyperHeuristic
    h = SelectionHyperHeuristic(
        optimizers=optimizers,
        selection_strategy=strategy,
        performance_metric=performance_metric,
        selection_interval=10,
    )

    # Creates an Opytimizer object
    opt = Opytimizer(space, h, function)

    # Runs the hyperheuristic optimization
    opt.start(n_iterations=50)

    # Prints out the best agent found
    print(f"Best agent position: {opt.space.best_agent.position}")
    print(f"Best agent fitness: {opt.space.best_agent.fit}")
    print(f"Optimization time: {opt.history.time} seconds")

    # Print optimizer selection statistics
    stats = h.get_strategy_statistics()
    print(f"Optimizer selections: {stats['optimizer_selections']}")
