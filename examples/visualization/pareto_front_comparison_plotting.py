import numpy as np


from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary import RVEA, MOEAD
from opytimizer.visualization import  plot_pareto_front_comparison
from opytimizer.utils.reference_vectors import das_dennis
from opytimizer import Opytimizer
from opytimizer.utils.operators import PolynomialMutation, SBXCrossover

N_GENERATIONS = 100

N_VARIABLES = 30
N_OBJECTIVES = 2
LOWER_BOUND = [0.] * N_VARIABLES
UPPER_BOUND = [1.] * N_VARIABLES

WEIGHTS, N_AGENTS = das_dennis(n_objectives=N_OBJECTIVES, n_partitions=99)



def zdt1(x: np.ndarray) -> np.ndarray:
    x = x.flatten()
    n = len(x)
    
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1.0)
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    
    return np.array([f1, f2])


func = Function(zdt1)

np.random.seed(0)

#####################################################################################################################################
space = SearchSpace(n_agents=N_AGENTS, n_variables=N_VARIABLES, n_objectives=N_OBJECTIVES, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)
optimizer = RVEA(reference_vectors=WEIGHTS, mutation_operator=PolynomialMutation(eta=20, rate=(1./N_VARIABLES)), crossover_operator=SBXCrossover(eta=30, rate=1.0, gene_rate=1.0, return_mode='both'), max_generations=N_GENERATIONS)

opt_rvea = Opytimizer(space=space, optimizer=optimizer, function=func, save_agents=True)

opt_rvea.start(N_GENERATIONS)
#####################################################################################################################################

np.random.seed(0)

#####################################################################################################################################

space = SearchSpace(n_agents=N_AGENTS, n_variables=N_VARIABLES, n_objectives=N_OBJECTIVES, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)
optimizer = MOEAD(weights_vector=WEIGHTS, mutation_operator=PolynomialMutation(eta=20, rate=(1./N_VARIABLES)), crossover_operator=SBXCrossover(eta=30, rate=1.0, gene_rate=1.0, return_mode='random'))

opt_moead = Opytimizer(space=space, optimizer=optimizer, function=func, save_agents=True)

opt_moead.start(N_GENERATIONS)

#####################################################################################################################################

plot_pareto_front_comparison(opt_rvea.space.pareto_front, opt_moead.space.pareto_front, backend="matplotlib", title="RVEA x MOEA/D - ZDT1",labels=['RVEA', 'MOEA/D']).show()