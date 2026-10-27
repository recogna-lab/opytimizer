---
sidebar_position: 15
title: Custom Callbacks
---

# Custom Callbacks

Learn how to interact with the optimization pipeline by creating your own callbacks.

## How It Works

A custom callback inherits from `Callback` and overrides only the hooks it needs. Callbacks are passed to the `start()` method as a list and are called in the given order.

## Example Code

```python
import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import Callback


class BestFitnessCallback(Callback):
    """A BestFitnessCallback class that records the best fitness
    found at the end of each iteration.

    """

    def __init__(self):
        super(BestFitnessCallback, self).__init__()

        self.fitness = []

    def on_task_begin(self, opt_model):
        """Called at the beginning of the task."""

        self.fitness = []

    def on_iteration_end(self, iteration, opt_model):
        """Called at the end of an iteration."""

        self.fitness.append(opt_model.space.best_agent.fit)


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

# Runs the optimization task with the custom callback
callback = BestFitnessCallback()
opt.start(MaxIterations(100), callbacks=[callback])

print(f"Best fitness at iterations 1, 50 and 100: {callback.fitness[0]}, {callback.fitness[49]}, {callback.fitness[-1]}")
```

## Available Hooks

* **`on_task_begin(opt_model)`** and **`on_task_end(opt_model)`**
* **`on_iteration_begin(iteration, opt_model)`** and **`on_iteration_end(iteration, opt_model)`**
* **`on_evaluate_before(*evaluate_args)`** and **`on_evaluate_after(*evaluate_args)`**
* **`on_update_before(*update_args)`** and **`on_update_after(*update_args)`**
