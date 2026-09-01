from opytimark.markers.n_dimensional import Sphere
from opytimizer.core.stopping import MaxIterations
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.hyperheuristics.adaptation_mechanism import (
    ParameterAdaptation,
    PopulationAdaptation,
    StrategyAdaptation,
)
from opytimizer.hyperheuristics.adaptive import AdaptiveHyperHeuristic
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace


# Define a performance metric function
def performance_metric(space):
    """Calculate performance based on best fitness value."""
    return space.best_agent.fit


# For adaptive hyperheuristic, we only need ONE optimizer
# The adaptive mechanism will modify its parameters/strategies
optimizers = [PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7})]

# One should declare different adaptation mechanisms to compare
adaptation_mechanisms = {
    "Parameter Adaptation": ParameterAdaptation(),
    "Strategy Adaptation": StrategyAdaptation(),
    "Population Adaptation": PopulationAdaptation(),
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

# Test different adaptation mechanisms
for mechanism_name, mechanism in adaptation_mechanisms.items():
    print(f"\n=== Testing {mechanism_name} ===")

    # Creates an AdaptiveHyperHeuristic
    h = AdaptiveHyperHeuristic(
        optimizers=optimizers,
        adaptation_mechanism=mechanism,
        performance_metric=performance_metric,
        adaptation_interval=15,
    )

    # Creates an Opytimizer object
    opt = Opytimizer(space, h, function)

    # Runs the hyperheuristic optimization
    opt.start(MaxIterations(60))

    # Prints out the best agent found
    print(f"Best agent position: {opt.space.best_agent.position}")
    print(f"Best agent fitness: {opt.space.best_agent.fit}")
    print(f"Optimization time: {opt.history.time} seconds")

    # Print adaptation statistics
    stats = h.get_adaptation_statistics()
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Adaptation mechanism: {stats['adaptation_mechanism']}")
    print(f"Adaptation interval: {stats['adaptation_interval']}")
    print(f"Adaptations performed: {len(stats['adaptation_history'])}")
