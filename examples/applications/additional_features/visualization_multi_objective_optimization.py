import numpy as np
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.multi_objective import NSGA2
from opytimizer.spaces import SearchSpace
from opytimizer.visualization.multi_objective import (
    plot_pareto_front,
    plot_pareto_evolution
)

from opytimizer.utils.operators import sbx_crossover, polynomial_mutation
from opytimizer.utils.callback import Callback

# Random seed for experimental consistency
np.random.seed(0)

# ZDT1 function
def zdt1(x):
    # Calculates f1 and f2
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    # Avoids division by zero
    if f1 == 0:
        f2 = g
    else:
        f2 = g * (1 - np.sqrt(f1/g))
    
    # Returns a 1D array with the two objectives
    return np.array([f1, f2], dtype=np.float64)

# Number of agents and decision variables
n_agents = 100
n_variables = 30

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0] * n_variables
upper_bound = [1] * n_variables

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = NSGA2(
    crossover_operator=sbx_crossover,
    mutation_operator=polynomial_mutation,
    crossover_params={'eta': 20},
    mutation_params={'eta': 20}
)
function = Function(zdt1)

# List to store the Pareto fronts at different iterations
pareto_fronts = []
iterations = [1, 50, 100, 150, 200, 250]

class ParetoFrontSaver(Callback):
    def __init__(self, optimizer, pareto_fronts, iterations):
        super().__init__()
        self.optimizer = optimizer
        self.pareto_fronts = pareto_fronts
        self.iterations = iterations

    def on_iteration_end(self, iteration, opt_model):
        if iteration in self.iterations:
            if hasattr(self.optimizer, 'pareto_front') and self.optimizer.pareto_front:
                print(f"Salvando frente na iteração {iteration} - tamanho: {len(self.optimizer.pareto_front)}")
                self.pareto_fronts.append(self.optimizer.pareto_front.copy())
            else:
                print(f"Pareto front in iteration {iteration} is empty.")

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization passing the callback
opt.start(n_iterations=250, callbacks=[ParetoFrontSaver(optimizer, pareto_fronts, iterations)])

# Plots the final Pareto front
plot_pareto_front(
    optimizer.pareto_front,
    all_solutions=space.agents,
    title="ZDT1 Pareto Front",
    subtitle="NSGA-II",
    xlabel="f1",
    ylabel="f2"
)

# Plots the evolution of the Pareto front
plot_pareto_evolution(
    pareto_fronts,
    iterations[:len(pareto_fronts)],
    title="ZDT1 Pareto Front Evolution",
    subtitle="NSGA-II"
)