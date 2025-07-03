"""Example demonstrating the use of PerformanceTrackingCallback with hyperheuristics.
This shows how the callback system can be used for performance tracking instead of
a separate PerformanceTracker class.
"""

import time

from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.hyperheuristics.selection import SelectionHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import ChoiceFunction
from opytimizer.optimizers.single_objective.swarm import ABC, PSO
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import PerformanceTrackingCallback


# Define a performance metric function
def performance_metric(space):
    """Calculate performance based on best fitness value."""
    return space.best_agent.fit


# Create optimizers
optimizers = [
    PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}),
    ABC(params={"n_trials": 10}),
]

# Create selection strategy
selection_strategy = ChoiceFunction()

# Create performance tracking callback
performance_callback = PerformanceTrackingCallback(window_size=5)

# Create hyperheuristic
h = SelectionHyperHeuristic(
    optimizers=optimizers,
    selection_strategy=selection_strategy,
    performance_metric=performance_metric,
    selection_interval=10,
)

# Create function and space
function = Function(Sphere())
space = SearchSpace(
    n_agents=20,
    n_variables=2,
    n_objectives=1,
    lower_bound=[-10, -10],
    upper_bound=[10, 10],
)

# Create Opytimizer
opt = Opytimizer(space, h, function)

# Run optimization with performance tracking callback
print("Starting optimization with PerformanceTrackingCallback...")
start_time = time.time()

opt.start(n_iterations=50, callbacks=[performance_callback])

total_time = time.time() - start_time

# Get statistics from the callback
stats = performance_callback.get_statistics()

print("\n=== PERFORMANCE TRACKING CALLBACK STATISTICS ===")
print(f"Performance history: {stats['performance_history']}")
print(f"Selection history: {stats['selection_history']}")
print(f"Performance ranking: {stats['performance_ranking']}")

print("\n=== OPTIMIZER ANALYSIS ===")
for optimizer_name, optimizer_stats in stats["optimizer_statistics"].items():
    print(f"\n{optimizer_name}:")
    print(f"  - Best performance: {optimizer_stats['best_performance']}")
    print(f"  - Average performance: {optimizer_stats['average_performance']}")
    print(f"  - Selection frequency: {optimizer_stats['selection_frequency']:.2%}")

print("\n=== FINAL RESULTS ===")
print(f"Best agent position: {opt.space.best_agent.position}")
print(f"Best agent fitness: {opt.space.best_agent.fit}")
print(f"Total optimization time: {total_time:.4f} seconds")

# Compare with hyperheuristic's own statistics
print("\n=== COMPARISON WITH HYPERHEURISTIC STATISTICS ===")
h_stats = h.get_statistics()
print(f"Hyperheuristic selections: {h_stats['optimizer_selections']}")
print(f"Hyperheuristic best performances: {h_stats['best_performances']}")

# Show that both approaches give similar results
print("\n=== VALIDATION ===")
print("Both the callback and hyperheuristic track the same information:")
for optimizer_name in stats["optimizer_statistics"]:
    callback_best = stats["optimizer_statistics"][optimizer_name]["best_performance"]
    h_best = h_stats["best_performances"].get(optimizer_name)
    print(f"{optimizer_name}: Callback={callback_best}, Hyperheuristic={h_best}")
