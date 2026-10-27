---
sidebar_position: 14
title: Creating a Custom Optimizer
---

# Creating a Custom Optimizer

Learn how to implement your own meta-heuristic by inheriting from the `Optimizer` class.

## How It Works

Every optimizer follows the same contract:

* **`__init__`**: Declares the default hyperparameters and calls `self.build(params)`, which overrides them with user-defined values.
* **`compile(space)`**: (Optional) Creates additional variables that depend on the space, such as velocities or memories.
* **`update(space)`**: Updates the agents' positions at each iteration.
* **`evaluate(space, function)`**: (Optional) Evaluates the agents. The default implementation evaluates every agent and updates the best agent.

After each update, `Opytimizer` automatically clips the agents to the space bounds.

## Example Code

```python
import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function, Optimizer
from opytimizer.core.stopping import MaxIterations
from opytimizer.spaces import SearchSpace


class RandomWalk(Optimizer):
    """A RandomWalk class, inherited from Optimizer.

    Each agent takes a random step towards the best agent.

    """

    def __init__(self, params=None):
        super(RandomWalk, self).__init__()

        # Default hyperparameters
        self.step_size = 0.5

        # Overrides the default hyperparameters with `params`
        self.build(params)

    def compile(self, space):
        # Creates a variable that depends on the space
        self.n_steps = np.zeros(space.n_agents)

    def update(self, space):
        for i, agent in enumerate(space.agents):
            # Random step towards the best agent
            r1 = np.random.uniform(size=agent.position.shape)
            agent.position += self.step_size * r1 * (space.best_agent.position - agent.position)

            # Random perturbation
            agent.position += np.random.normal(0, 0.1, size=agent.position.shape)

            self.n_steps[i] += 1


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
optimizer = RandomWalk(params={"step_size": 0.8})
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(MaxIterations(100))
```
