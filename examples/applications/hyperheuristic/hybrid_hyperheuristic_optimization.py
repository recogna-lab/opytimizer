from opytimark.markers.n_dimensional import Sphere
from opytimizer.core.stopping import MaxIterations
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.hyperheuristics.adaptation_mechanism import (
    ParameterAdaptation,
    StrategyAdaptation,
)
from opytimizer.hyperheuristics.hybrid import HybridHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import (
    ChoiceFunction,
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

# One should declare different combinations of selection and adaptation
combinations = [
    {
        "name": "Choice Function + Parameter Adaptation",
        "selection": ChoiceFunction(),
        "adaptation": ParameterAdaptation(),
    },
    {
        "name": "Multi-Armed Bandit + Strategy Adaptation",
        "selection": MultiArmedBandit(),
        "adaptation": StrategyAdaptation(),
    },
]

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

# Test different hybrid combinations
for combo in combinations:
    print(f"\n=== Testing {combo['name']} ===")

    # Creates a HybridHyperHeuristic
    h = HybridHyperHeuristic(
        optimizers=optimizers,
        selection_strategy=combo["selection"],
        adaptation_mechanism=combo["adaptation"],
        performance_metric=performance_metric,
        selection_interval=8,
        adaptation_interval=12,
    )

    # Creates an Opytimizer object
    opt = Opytimizer(space, h, function)

    # Runs the hyperheuristic optimization
    opt.start(MaxIterations(80))

    # Prints out the best agent found
    print(f"Best agent position: {opt.space.best_agent.position}")
    print(f"Best agent fitness: {opt.space.best_agent.fit}")
    print(f"Optimization time: {opt.history.time} seconds")

    # Print hybrid statistics
    stats = h.get_hybrid_statistics()
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Optimizer selections: {stats['optimizer_selections']}")
    print(f"Selection strategy: {stats['selection_strategy']}")
    print(f"Adaptation mechanism: {stats['adaptation_mechanism']}")
