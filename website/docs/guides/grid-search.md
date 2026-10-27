---
sidebar_position: 10
title: Grid Search
---

# Grid Search

Learn how to exhaustively evaluate a discretized search space using `GridSpace` and the `GS` optimizer.

## How It Works

`GridSpace` builds a grid from the `step` size of each decision variable, and each grid point becomes an agent. Hence, the number of agents is defined by the grid, not by the user.

Since every point is evaluated at once, a single iteration is enough to find the best grid point.

## Example Code

```python
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.misc import GS
from opytimizer.spaces import GridSpace

# Number of decision variables and objectives
n_variables = 2
n_objectives = 1

# Step size of each variable (has to be the same size as `n_variables`)
step = [0.1, 1]

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = GridSpace(n_variables, n_objectives, step, lower_bound, upper_bound)
optimizer = GS()
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(MaxIterations(1))

# Prints out the number of evaluated grid points and the best one
print(f"Grid points: {space.n_agents} | Best Agent: {opt.space.best_agent.position.ravel()}")
```
