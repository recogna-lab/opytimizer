import numpy as np


from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.evolutionary import DE
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.optimizers.single_objective.population import GWO
from opytimizer.visualization import plot_convergence
from opytimizer import Opytimizer

from opytimark.markers.n_dimensional import Sphere

np.random.seed(0)

N_AGENTS = 20
N_GENERATIONS = 250

n_variables = 2
n_objectives = 1
lower_bound = [-10, -10]
upper_bound = [10, 10]

func = Function(Sphere())

space = SearchSpace(n_agents=N_AGENTS, n_variables=n_variables, n_objectives=n_objectives, lower_bound=lower_bound, upper_bound=upper_bound)
de = DE()
opt_de = Opytimizer(space=space, optimizer=de, function=func, save_agents=False)
opt_de.start(N_GENERATIONS)

#############################################################################################
np.random.seed(0)
#############################################################################################


space = SearchSpace(n_agents=N_AGENTS, n_variables=n_variables, n_objectives=n_objectives, lower_bound=lower_bound, upper_bound=upper_bound)
pso = PSO()
opt_pso = Opytimizer(space=space, optimizer=pso, function=func, save_agents=False)
opt_pso.start(N_GENERATIONS)


#############################################################################################
np.random.seed(0)
#############################################################################################

space = SearchSpace(n_agents=N_AGENTS, n_variables=n_variables, n_objectives=n_objectives, lower_bound=lower_bound, upper_bound=upper_bound)
gwo = GWO()
opt_gwo = Opytimizer(space=space, optimizer=gwo, function=func, save_agents=False)
opt_gwo.start(N_GENERATIONS)


plot_convergence(opt_de.history.best_agent, opt_pso.history.best_agent, opt_gwo.history.best_agent, labels=['DE', 'PSO', 'GWO'], backend='matplotlib').show()