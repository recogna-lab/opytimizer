import numpy as np
from opytimark.markers.n_dimensional import Rastrigin, Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.spaces import SearchSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents, decision variables and objectives
n_agents = 20
n_variables = 2
n_objectives = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = NSGA2()
function = Function([Rastrigin(), Sphere()])

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=1000)