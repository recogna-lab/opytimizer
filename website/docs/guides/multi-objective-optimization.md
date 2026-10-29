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
from opytimizer.benchmarks import ZDT3
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.spaces import SearchSpace

# Random seed for experimental consistency
np.random.seed(0)

# ZDT3 benchmark problem (30 decision variables, 2 objectives)
problem = ZDT3(n_variables=30)

# Number of agents
n_agents = 100

# Creates the space from the problem's bounds, the optimizer and the function
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
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(MaxIterations(1000))
```

The true Pareto front of the problem is available through `problem.pareto_front()`, which can be compared with the front found by the optimizer (`opt.space.pareto_front`). See [Benchmark Problems](benchmark-problems.md) for other problems and for computing metrics against the true front.
