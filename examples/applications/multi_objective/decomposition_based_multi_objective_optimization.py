import numpy as np
from opytimizer import Opytimizer
from opytimizer.core.stopping import MaxIterations
from opytimizer.core import Function
from opytimizer.optimizers.multi_objective.evolutionary import MOEAD
from opytimizer.spaces import SearchSpace
from opytimizer.utils.reference_vectors import das_dennis
from opytimizer.math.aggregation import PBI

def zdt3(x: np.ndarray) -> np.ndarray:
    """
        ZDT3 benchmark problem
        References:
        Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
        IEEE Transactions on evolutionary computation, 11(6), 712-731.

    """
    
    # x have (n, 1) shape, where n is the number of decision variables
    x = x.flatten()
    f1 = x[0]
    n = x.shape[0]
    g = 1 + (9 * np.sum(x[1:])) / (n - 1)
    f2 = g * (1 - np.sqrt(f1/g) - (f1/g) * np.sin(10 * np.pi * x[0]))
    
    return [f1, f2]
# Random seed for experimental consistency
np.random.seed(0)

n_variables = 30
n_objectives = 2
weights, n_agents = das_dennis(2, 99)
# Number of agents, decision variables and objectives

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0] * n_variables
upper_bound = [1] * n_variables

pbi = PBI()

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = MOEAD(weight_vectors=weights,decomposition_method=pbi)
function = Function(zdt3)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(MaxIterations(250))