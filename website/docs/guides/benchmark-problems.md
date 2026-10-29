---
sidebar_position: 1.5
title: Benchmark Problems
---

# Benchmark Problems

Learn how to evaluate optimizers on the built-in benchmark problems of `opytimizer.benchmarks`: classic single-objective functions, the CEC 2022 suite and the ZDT, DTLZ, WFG and ZCAT multi- and many-objective families. See [Benchmark Problems](../concepts/benchmarks.md) for the full list of problems and their properties.

## Single-Objective Functions

A benchmark is a callable, so it is wrapped by `Function` like any objective function. Its search domain and its optimum are available as attributes, which makes it easy to measure the error of a run.

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.benchmarks import Rastrigin
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace

# Random seed for experimental consistency
np.random.seed(0)

# Creates a 10-dimensional Rastrigin's function on its standard domain
problem = Rastrigin(n_variables=10)

# Creates the space from the problem's bounds
space = SearchSpace(
    n_agents=30,
    n_variables=problem.n_variables,
    n_objectives=1,
    lower_bound=problem.lower_bound,
    upper_bound=problem.upper_bound,
)
optimizer = PSO()
function = Function(problem)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(MaxIterations(500))

# Error with respect to the known global optimum
error = float(opt.space.best_agent.fit) - problem.optimal_value
print(f"Best fitness: {float(opt.space.best_agent.fit):.4f} | Error: {error:.4f}")
```

The standard domain can be replaced through `lower_bound` and `upper_bound`, e.g., `Rastrigin(n_variables=10, lower_bound=-10.0, upper_bound=10.0)`.

## CEC 2022 Suite

The CEC 2022 functions keep their official biases, so the error of a run is the difference between the best fitness and `optimal_value`. They are available for $D \in \{2, 10, 20\}$ ($D \in \{10, 20\}$ for F6-F8).

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.benchmarks import CEC2022F1, CEC2022F4, CEC2022F12
from opytimizer.core import Function
from opytimizer.core.stopping import MaxEvaluations
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace

np.random.seed(0)

for problem_class in (CEC2022F1, CEC2022F4, CEC2022F12):
    problem = problem_class(n_variables=10)

    space = SearchSpace(
        30, problem.n_variables, 1, problem.lower_bound, problem.upper_bound
    )

    # Logging and progress bars are disabled, since many runs are executed
    opt = Opytimizer(
        space,
        PSO(),
        Function(problem),
        enable_logging=False,
        show_progress=False,
    )
    opt.start(MaxEvaluations(20000))

    error = float(opt.space.best_agent.fit) - problem.optimal_value
    print(f"{problem.name}: error = {error:.4e}")
```

## Multi-Objective Problems and Their Pareto Fronts

Every multi-objective problem returns a sample of its true Pareto front through `pareto_front()`, which can be used as the reference set of `IGD`.

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.benchmarks import ZDT1
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.math.metrics import IGD
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.spaces import SearchSpace
from opytimizer.utils.operators import PolynomialMutation

np.random.seed(0)

# ZDT1 with the 30 variables recommended in the literature
problem = ZDT1()

space = SearchSpace(
    100,
    problem.n_variables,
    problem.n_objectives,
    problem.lower_bound,
    problem.upper_bound,
)
optimizer = NSGA2(mutation_operator=PolynomialMutation(rate=1 / problem.n_variables))
function = Function(problem)

opt = Opytimizer(space, optimizer, function, save_history=True)

# The true front of the problem is the reference set of IGD
opt.start(MaxIterations(250), metrics=[IGD(problem.pareto_front(500))])

print(f"IGD: {opt.history.igd[-1]:.4f}")
```

## Many-Objective Problems

DTLZ, WFG and ZCAT scale to any number of objectives. When the number of variables is omitted, the value recommended in the literature is used.

```python
from opytimizer.benchmarks import DTLZ2, WFG4, ZCAT12

dtlz2 = DTLZ2(n_objectives=5)                       # 5 + 10 - 1 = 14 variables
wfg4 = WFG4(n_objectives=5)                         # k = 8 and l = 20, i.e., 28 variables
zcat12 = ZCAT12(n_variables=30, n_objectives=5, level=3, bias=True)

for problem in (dtlz2, wfg4, zcat12):
    front = problem.pareto_front(n_points=1000)
    print(f"{problem.name}: {problem.n_variables} variables, front {front.shape}")
```

## GPU Execution

A benchmark runs on the backend of its `env`. With a CuPy environment and a tensorized space, the whole population is evaluated on the GPU at once, and the Pareto front is also returned as a CuPy array, which metrics accept directly.

```python
from opytimizer import Opytimizer
from opytimizer.benchmarks import DTLZ1
from opytimizer.core import Environment, Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.math.metrics import IGD
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2Tensor
from opytimizer.spaces import SearchSpace
from opytimizer.utils.operators import PolynomialMutationTensor

env = Environment("cupy", "float32")

# The problem and the space share the same environment
problem = DTLZ1(n_variables=10, n_objectives=3, env=env)

space = SearchSpace(
    300,
    problem.n_variables,
    problem.n_objectives,
    problem.lower_bound,
    problem.upper_bound,
    env=env,
    tensorized=True,
)
optimizer = NSGA2Tensor(
    mutation_operator=PolynomialMutationTensor(env=env, rate=1 / problem.n_variables)
)

opt = Opytimizer(space, optimizer, Function(problem), save_history=True)

# The reference front lives on the GPU as well
front = problem.pareto_front(1000)
opt.start(MaxIterations(250), metrics=[IGD(front)], metrics_every=50)

print(f"IGD: {opt.history.igd[-1]:.4f}")
```

:::note
Metrics run independently of the environment: GPU fronts and references are copied to the host once per evaluation and the metric is computed with NumPy. Use `metrics_every` to evaluate them only every few iterations.
:::

:::tip
Use a NumPy environment (the default) with standard, agent-by-agent optimizers, and a CuPy environment with tensorized (`*Tensor`, `*Cuda`) optimizers.
:::

## Running a Benchmark Study

Since problems, optimizers and runners are independent objects, a study over several problems and seeds is a pair of loops. Disabling logging and progress bars keeps the output clean.

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.benchmarks import WFG1, WFG4, WFG9
from opytimizer.core import Environment, Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.math.metrics import IGD
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2Tensor
from opytimizer.spaces import SearchSpace
from opytimizer.utils.operators import PolynomialMutationTensor

env = Environment("numpy", "float64")
results = {}

for problem_class in (WFG1, WFG4, WFG9):
    problem = problem_class(n_objectives=2, env=env)
    igd = IGD(problem.pareto_front(500))

    for seed in range(3):
        np.random.seed(seed)

        space = SearchSpace(
            100,
            problem.n_variables,
            problem.n_objectives,
            problem.lower_bound,
            problem.upper_bound,
            env=env,
            tensorized=True,
        )
        optimizer = NSGA2Tensor(
            mutation_operator=PolynomialMutationTensor(
                env=env, rate=1 / problem.n_variables
            )
        )

        opt = Opytimizer(
            space,
            optimizer,
            Function(problem),
            enable_logging=False,
            show_progress=False,
        )
        opt.start(MaxIterations(200))

        results.setdefault(problem.name, []).append(igd(opt.space.pareto_front))

for name, values in results.items():
    print(f"{name}: IGD = {np.mean(values):.4f} ± {np.std(values):.4f}")
```

## Evaluating a Problem Directly

Benchmarks can also be called outside of an optimization task, with a single solution or a batch of them:

```python
import numpy as np

from opytimizer.benchmarks import DTLZ2

problem = DTLZ2(n_variables=12, n_objectives=3)

problem(np.full(12, 0.5))         # one solution -> shape (3,)
problem(np.random.rand(100, 12))  # a batch -> shape (100, 3)
```
