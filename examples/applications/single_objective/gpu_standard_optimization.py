import cupy as cp # Ensure that you have cupy installed

from opytimizer.core import  Function, Environment
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.swarm import PSOCuda
from opytimizer import Opytimizer

from opytimizer.core import  NoImprovement


def no_shifted_sphere(x):
    return cp.sum(cp.square(x), axis=1)

n_agents = 60
n_variables = 30
n_objectives = 1

lower_bound = [-30.0] * n_variables
upper_bound = [30.0] * n_variables

device_gpu = Environment().set_backend('cupy').set_dtype('float32')

my_search_space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound, env=device_gpu, tensorized=True)

stopping_criteria = NoImprovement(patience=10, min_delta=1e-5)
my_func = Function(no_shifted_sphere)

my_pso = PSOCuda()

opt = Opytimizer(space=my_search_space, optimizer=my_pso, function=my_func)

opt.start(stopping_criteria=stopping_criteria)
