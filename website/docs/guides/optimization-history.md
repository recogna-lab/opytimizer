---
sidebar_position: 17
title: Inspecting the Optimization History
---

# Inspecting the Optimization History

Learn how to store and inspect the evolution of the agents throughout the optimization task.

## How It Works

When `save_history=True` is passed to `Opytimizer`, the state of the task is dumped into `opt.history` at the end of each iteration. Every agent of the population is only stored when `save_agents=True` is also passed. Every key is stored as a list, where the last dimension stands for the iteration.

The `get_convergence()` method reshapes these lists into arrays of positions and fitness values.

## Example Code

```python
import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 3
n_variables = 2
n_objectives = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = PSO()
function = Function(Sphere())

# Bundles every piece into Opytimizer class
# `save_history` stores the task's state at every iteration,
# while `save_agents` also stores every agent (not only the best one)
opt = Opytimizer(space, optimizer, function, save_agents=True, save_history=True)

# Runs the optimization task
opt.start(MaxIterations(10))

# Gathers the best agent's convergence
best_agent_pos, best_agent_fit = opt.history.get_convergence("best_agent")
print(f"Last iteration - Best agent (position, fit): ({best_agent_pos[:, -1]}, {best_agent_fit[-1]})")
print(f"Iteration 4 - Best agent (position, fit): ({best_agent_pos[:, 3]}, {best_agent_fit[3]})")

# Gathers a specific agent's convergence
agent_0_pos, agent_0_fit = opt.history.get_convergence("agents", index=0)
print(f"Last iteration - Agent[0] (position, fit): ({agent_0_pos[:, -1]}, {agent_0_fit[-1]})")

# Prints out the optimization time
print(f"Optimization time: {opt.history.time[-1]} seconds")
```

## Custom History

The `History` class can also be used on its own. Any variable dumped into it is appended to a list:

```python
from opytimizer.utils.history import History

# Instantiates the History
h = History()

# Dumps a variable (it will be converted into a list)
h.dump(x=1)
h.dump(x=2)
h.dump(x=3)

# Accesses the variable
print(h.x)
```
