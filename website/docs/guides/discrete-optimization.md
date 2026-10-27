---
sidebar_position: 9
title: Discrete Optimization
---

# Discrete Optimization

Learn how to restrict continuous decision variables to a set of discrete values using `DiscreteSearchCallback`.

## How It Works

Before each evaluation, `DiscreteSearchCallback` maps every decision variable to its closest allowed value. Therefore, any continuous optimizer can be used to solve discrete problems.

The `allowed_values` argument must contain one list per decision variable, and every value must lie within the variable's bounds.

## Example Code

```python
import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import DiscreteSearchCallback

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 20
n_variables = 2
n_objectives = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Defines the allowed values for performing the discrete search
allowed_values = [list(range(lb, ub, 2)) for lb, ub in zip(lower_bound, upper_bound)]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = PSO()
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task with the discrete search callback
opt.start(
    MaxIterations(100), callbacks=[DiscreteSearchCallback(allowed_values=allowed_values)]
)
```
