import cupy as cp

from opytimizer.core import Environment, Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary import  MOEADTensor
from opytimizer.utils.operators import PolynomialMutationTensor, SBXCrossoverTensor
from opytimizer.core.stopping import MaxIterations
from opytimizer import Opytimizer
from opytimizer.utils.reference_vectors import das_dennis
from opytimizer.visualization.visualizer import  plot_agents, plot_pareto_front_evolution
from opytimizer.math.aggregation import PBI

def dtlz1(x: cp.ndarray) -> cp.ndarray:
    k = x[:, 2:].shape[1]
    g = 100.0 * (k + cp.sum((x[:, 2:] - 0.5) ** 2 - cp.cos(20.0 * cp.pi * (x[:, 2:] - 0.5)), axis=1))
    
    f1 = 0.5 * x[:, 0] * x[:, 1] * (1.0 + g)
    f2 = 0.5 * x[:, 0] * (1.0 - x[:, 1]) * (1.0 + g)
    f3 = 0.5 * (1.0 - x[:, 0]) * (1.0 + g)
    
    return cp.column_stack((f1, f2, f3))

SEED = 48
cp.random.seed(SEED)

gpu_env = Environment('cupy', 'float32')
N_VARS = 10
N_OBJS = 3
WEIGHTS, N_AGENTS = das_dennis(N_OBJS, 23)
MAX_GEN = 250
LB = [0.0] * N_VARS
UB = [1.0] * N_VARS
func = Function(dtlz1)

space = SearchSpace(n_agents=N_AGENTS,
                    n_variables=N_VARS,
                    n_objectives=N_OBJS,
                    lower_bound=LB,
                    upper_bound=UB,
                    env=gpu_env,
                    tensorized=True)

crossover = SBXCrossoverTensor(env=gpu_env)
mutation = PolynomialMutationTensor(rate=1/N_VARS, env=gpu_env)

opt = MOEADTensor(crossover_operator=crossover,
                  mutation_operator=mutation,
                  weight_vectors=WEIGHTS,
                  decomposition_method=PBI(),)

opy = Opytimizer(space=space,
                 optimizer=opt,
                 function=func,
                 save_agents=False,
                 save_history=True)


stop_criterion = MaxIterations(MAX_GEN)

opy.start(stopping_criteria=stop_criterion)

plot_agents(opy.space.pareto_front, backend='plotly').show()
plot_pareto_front_evolution(opy.history.pareto_front, iterations=[150, 200, 250],
                            title='Pareto Front Evolution - MOEA/D on DTLZ1 Benchmark Problem',
                            backend='matplotlib'
                            ).show()
