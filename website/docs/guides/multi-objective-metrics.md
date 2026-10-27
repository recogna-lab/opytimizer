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

## Example Code

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.math.metrics import HV, IGD
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.spaces import SearchSpace


def zdt1(x: np.ndarray) -> np.ndarray:
    x = x.flatten()
    n = len(x)

    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1.0)
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h

    return np.array([f1, f2])


# Random seed for experimental consistency
np.random.seed(0)

# Number of agents, decision variables and objectives
n_agents = 100
n_variables = 30
n_objectives = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0.0] * n_variables
upper_bound = [1.0] * n_variables

# Samples the true Pareto front of ZDT1, used as reference by IGD
f1 = np.linspace(0, 1, 500)
true_front = np.column_stack((f1, 1 - np.sqrt(f1)))

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = NSGA2()
function = Function(zdt1)

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
