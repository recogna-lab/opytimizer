---
sidebar_position: 3
title: Multi-Objective Optimization
---

# Multi-Objective Optimization

This guide covers standard multi-objective algorithms (such as NSGA-II) and decomposition-based algorithms (such as MOEA/D).

## Standard Multi-Objective Optimization (NSGA-II)

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.spaces import SearchSpace


def zdt3(x: np.ndarray) -> np.ndarray:
    """
    ZDT3 benchmark problem
    References:
    Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
    IEEE Transactions on evolutionary computation, 11(6), 712-731.
    """
    x = x.flatten()
    f1 = x[0]
    n = x.shape[0]
    g = 1 + (9 * np.sum(x[1:])) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * x[0]))

    return [f1, f2]


# Random seed for experimental consistency
np.random.seed(0)

# Number of agents, decision variables and objectives
n_agents = 20
n_variables = 2
n_objectives = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = NSGA2()
function = Function(zdt3)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(MaxIterations(1000))