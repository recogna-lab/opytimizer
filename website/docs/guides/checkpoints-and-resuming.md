---
sidebar_position: 16
title: Checkpoints and Resuming
---

# Checkpoints and Resuming

Learn how to save, load, and resume optimization tasks using `CheckpointCallback`, `save()`, and `Opytimizer.load()`.

## Continuing a Task

Every call to `start()` continues the optimization from where it stopped. Note that the stopping criteria consider the total number of iterations, so the following lines achieve the same results as a 100-iteration run:

```python
opt.start(MaxIterations(50))
opt.start(MaxIterations(75))
opt.start(MaxIterations(100))
```

## Example Code

```python
import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import CheckpointCallback

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 20
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
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
# CheckpointCallback saves the task as `iter_<iteration>_checkpoint.pkl` every `frequency` iterations
opt.start(MaxIterations(10), callbacks=[CheckpointCallback(frequency=10)])

# Deletes the optimization object
del opt

# Loads the task from file and resumes it until the 35th iteration
opt = Opytimizer.load("iter_10_checkpoint.pkl")
opt.start(MaxIterations(35))

# Saves the final task to a custom file
opt.save("final_task.pkl")
```
