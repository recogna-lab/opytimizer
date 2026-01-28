import numpy as np

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.hyperheuristics.selection import SelectionHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import (
    ChoiceFunction,
    MultiArmedBandit,
)
from opytimizer.math.metrics import HypervolumeMetric
from opytimizer.optimizers.multi_objective.evolutionary import MOEAD, NSGA2
from opytimizer.spaces import SearchSpace
from opytimizer.utils.operators import PolynomialMutation, SBXCrossover
from opytimizer.utils.weights_vector import das_dennis


# Define a multi-objective function (ZDT1)
def zdt1_function(x):
    """ZDT1 multi-objective function."""
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return [f1, f2]


# Define a performance metric function for multi-objective optimization
REFERENCE_POINT = np.array(
    [1.1, 1.1]
)  # For ZDT1, all solutions should be dominated by this point


hv = HypervolumeMetric(REFERENCE_POINT)


# Generate weight vectors for MOEAD
weights, n_subproblems = das_dennis(n_objectives=2, n_partitions=12)

# One should declare a list of optimizer instances to be used by the hyperheuristic
optimizers = [
    NSGA2(
        crossover_operator=SBXCrossover(eta=20, return_mode='both'),
        mutation_operator=PolynomialMutation(eta=20),
    ),
    MOEAD(
        crossover_operator=SBXCrossover(eta=20),
        mutation_operator=PolynomialMutation(eta=20),
    ),
]
# Set weights_vector for MOEAD if needed
optimizers[1].weights_vector = weights

# One should declare different selection strategies to compare
selection_strategies = {
    "Choice Function": ChoiceFunction(),
    "Multi-Armed Bandit": MultiArmedBandit(),
}

# Creates a Function object
function = Function(zdt1_function)

# Creates a SearchSpace object for multi-objective optimization
# Use n_subproblems to ensure compatibility with MOEAD
space = SearchSpace(
    n_agents=n_subproblems,
    n_variables=30,
    n_objectives=2,
    lower_bound=[0] * 30,
    upper_bound=[1] * 30,
)

# Test different selection strategies
for strategy_name, strategy in selection_strategies.items():
    print(f"\n=== Testing {strategy_name} ===")

    # Creates a SelectionHyperHeuristic
    h = SelectionHyperHeuristic(
        optimizers=optimizers,
        selection_strategy=strategy,
        performance_metric=hv,
        selection_interval=15,
    )

    # Creates an Opytimizer object
    opt = Opytimizer(space, h, function)

    # Runs the hyperheuristic optimization
    opt.start(n_iterations=100)

    # Prints out multi-objective optimization results
    print(f"Number of Pareto solutions: {len(opt.space.pareto_front)}")
    print(f"Optimization time: {opt.history.time} seconds")
    print(f"Final Hypervolume: {hv(opt.space.pareto_front)}")

    # Print selection statistics
    stats = h.get_strategy_statistics()
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Optimizer selections: {stats['optimizer_selections']}")
    print(f"Selection strategy: {stats['selection_strategy']}")
