---
sidebar_position: 12
title: Multi-Objective Performance Metrics
---

# Multi-Objective Performance Metrics

Learn how to track the quality of the Pareto front during a multi-objective task using performance metrics, such as `HV` and `IGD`.

## How It Works

Metrics are passed to the `start()` method through the `metrics` argument. At each iteration, they are computed over the current Pareto front and displayed in the progress bar.

When `save_history=True`, each metric is also stored in the history under its lowercase name (e.g., `opt.history.hv` and `opt.history.igd`).

To reduce the overhead of long tasks, use `metrics_every` to compute the metrics only every few iterations, e.g., `opt.start(MaxIterations(100), metrics=[...], metrics_every=10)`.

Metrics work the same way with standard spaces, tensorized NumPy spaces and tensorized CuPy (GPU) spaces, and their reference fronts may be NumPy or CuPy arrays. On the GPU, the front is copied to the host once per evaluation and the metric is computed with NumPy.

## Example Code

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.benchmarks import ZDT1
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.math.metrics import HV, IGD
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.spaces import SearchSpace


# Random seed for experimental consistency
np.random.seed(0)

# ZDT1 benchmark problem (30 decision variables, 2 objectives)
problem = ZDT1(n_variables=30)

# Number of agents
n_agents = 100

# Samples the true Pareto front of ZDT1, used as reference by IGD
true_front = problem.pareto_front(n_points=500)

# Creates the space, optimizer and function
space = SearchSpace(
    n_agents,
    problem.n_variables,
    problem.n_objectives,
    problem.lower_bound,
    problem.upper_bound,
)
optimizer = NSGA2()
function = Function(problem)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_history=True)

# Runs the optimization task while tracking the metrics
opt.start(
    MaxIterations(100),
    metrics=[HV(reference_point=[1.1, 1.1]), IGD(true_front)],
)

# Prints out the metrics of the last iteration
print(f"HV: {opt.history.hv[-1]} | IGD: {opt.history.igd[-1]}")
```
