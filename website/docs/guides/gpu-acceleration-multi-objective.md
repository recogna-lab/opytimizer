---
sidebar_position: 6
title: GPU Multi-Objective Optimization (MOEA/D Tensor)
---

# Multi-Objective GPU Optimization (MOEA/D Tensor)

Learn how to leverage GPU execution via CuPy using vectorized operators and tensorized search spaces for multi-objective decomposition algorithms.

## Example Code

```python
import cupy as cp

from opytimizer import Opytimizer
from opytimizer.benchmarks import DTLZ1
from opytimizer.core import Environment, Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.math.aggregation import PBI
from opytimizer.optimizers.multi_objective.evolutionary import MOEADTensor
from opytimizer.spaces import SearchSpace
from opytimizer.utils.operators import PolynomialMutationTensor, SBXCrossoverTensor
from opytimizer.utils.reference_vectors import das_dennis
from opytimizer.visualization.visualizer import plot_agents, plot_pareto_front_evolution


SEED = 48
cp.random.seed(SEED)

gpu_env = Environment("cupy", "float32")

# DTLZ1 benchmark problem, evaluated on the GPU (10 variables, 3 objectives)
problem = DTLZ1(n_variables=10, n_objectives=3, env=gpu_env)
N_VARS = problem.n_variables
N_OBJS = problem.n_objectives
WEIGHTS, N_AGENTS = das_dennis(N_OBJS, 23)
MAX_GEN = 250
func = Function(problem)

space = SearchSpace(
    n_agents=N_AGENTS,
    n_variables=N_VARS,
    n_objectives=N_OBJS,
    lower_bound=problem.lower_bound,
    upper_bound=problem.upper_bound,
    env=gpu_env,
    tensorized=True,
)

crossover = SBXCrossoverTensor(env=gpu_env)
mutation = PolynomialMutationTensor(rate=1 / N_VARS, env=gpu_env)

opt = MOEADTensor(
    crossover_operator=crossover,
    mutation_operator=mutation,
    weight_vectors=WEIGHTS,
    decomposition_method=PBI(),
)

opy = Opytimizer(
    space=space, optimizer=opt, function=func, save_agents=False, save_history=True
)

stop_criterion = MaxIterations(MAX_GEN)

opy.start(stopping_criteria=stop_criterion)
```

The true Pareto front of DTLZ1, returned by `problem.pareto_front()` as a CuPy array, can be passed straight to a metric to track the quality of the front found on the GPU, e.g., `opy.start(stop_criterion, metrics=[IGD(problem.pareto_front())], metrics_every=10)` (with `from opytimizer.math.metrics import IGD`). See [Benchmark Problems](benchmark-problems.md) for more examples.
