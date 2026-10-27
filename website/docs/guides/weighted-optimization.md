---
sidebar_position: 11
title: Weighted Multi-Objective Optimization
---

# Weighted Multi-Objective Optimization

Learn how to solve multi-objective problems with single-objective optimizers by aggregating the objectives into a weighted sum using `WeightedFunction`.

## How It Works

`WeightedFunction` reduces $M$ objectives into a single scalar value, $f(x) = \sum_{i=1}^{M} w_i f_i(x)$. Thus, the space must be created with a single objective (`n_objectives = 1`).

Each set of weights leads to a single trade-off solution. To approximate the Pareto front, run the task with different weights or use a multi-objective optimizer instead.

## Example Code

```python
import numpy as np
from opytimark.markers.n_dimensional import Rastrigin, Sphere

from opytimizer import Opytimizer
from opytimizer.core.stopping import MaxIterations
from opytimizer.functions import WeightedFunction
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace

# Wraps the benchmark functions into named functions
rastrigin = Rastrigin()
sphere = Sphere()


def f1(x):
    return rastrigin(x)


def f2(x):
    return sphere(x)


# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
# Note that the aggregated function has a single objective
n_agents = 20
n_variables = 2
n_objectives = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
# Each function is multiplied by its corresponding weight
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = PSO()
function = WeightedFunction([f1, f2], [0.5, 0.5])

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(MaxIterations(1000))
```
