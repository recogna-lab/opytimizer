import numpy as np

from opytimizer.core.stopping import MaxIterations
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.visualization import plot_population_distribution_histogram
from opytimizer import Opytimizer
from opytimizer.utils.operators import PolynomialMutation, SBXCrossover

def zdt1(x: np.ndarray) -> np.ndarray:
    x = x.flatten()
    n = len(x)
    
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1.0)
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    
    return np.array([f1, f2])


np.random.seed(0)

N_AGENTS = 100
N_GENERATIONS = 100

N_VARIABLES = 30
N_OBJECTIVES = 2
LOWER_BOUND = [0.] * N_VARIABLES
UPPER_BOUND = [1.] * N_VARIABLES

space = SearchSpace(n_agents=N_AGENTS, n_variables=N_VARIABLES, n_objectives=N_OBJECTIVES, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)
func = Function(zdt1)

optimizer = NSGA2( mutation_operator=PolynomialMutation(eta=20, rate=(1./N_VARIABLES)), crossover_operator=SBXCrossover(eta=30, rate=1.0, gene_rate=1.0, n_offspring=2))

opt = Opytimizer(space=space, optimizer=optimizer, function=func, save_agents=False)

opt.start(MaxIterations(N_GENERATIONS))

# Based on F1 (target)
plot_population_distribution_histogram(opt.space.pareto_front, backend="matplotlib", title="NSGA2 - Pareto Front", target=1, label='F_1 values').show()

