---
sidebar_position: 4
title: Decomposition-Based Multi-Objective Optimization (MOEA/D)
---

# Decomposition-Based Multi-Objective Optimization (MOEA/D)

This guide demonstrates how to solve multi-objective optimization problems by decomposing them into a set of single-objective subproblems using `MOEAD` and scalarization functions such as `PBI`.

## Example Code

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.benchmarks import ZDT3
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.math.aggregation import PBI
from opytimizer.optimizers.multi_objective.evolutionary import MOEAD
from opytimizer.spaces import SearchSpace
from opytimizer.utils.reference_vectors import das_dennis


# Random seed for experimental consistency
np.random.seed(0)

# ZDT3 benchmark problem (30 decision variables, 2 objectives)
problem = ZDT3(n_variables=30)
weights, n_agents = das_dennis(problem.n_objectives, 99)

pbi = PBI()

# Creates the space, optimizer and function
space = SearchSpace(
    n_agents,
    problem.n_variables,
    problem.n_objectives,
    problem.lower_bound,
    problem.upper_bound,
)
optimizer = MOEAD(weight_vectors=weights, decomposition_method=pbi)
function = Function(problem)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(MaxIterations(250))
```
