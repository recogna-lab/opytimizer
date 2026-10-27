---
sidebar_position: 6
title: Agents
---

# Agents

An agent is a single candidate solution of the search space. Every space holds a population of agents that is iteratively updated by the optimizer.

## Agent Properties

* **`position`**: Decision variables with shape `(n_variables, n_dimensions)`.
* **`fit`**: Fitness value (or objective vector, in multi-objective tasks) of the current position.
* **`lb`** and **`ub`**: Lower and upper bounds of each decision variable.
* **`mapping`**: String-based identifiers for each decision variable.
* **`mapped_position`**: Dictionary that maps each identifier to its position value.
* **`ts`**: Timestamp of the agent's last improvement.

## Dimensions

The `n_dimensions` property defines how many values represent a single decision variable:

* **`n_dimensions = 1`**: Real-valued (or boolean) variable.
* **`n_dimensions = 2`**: Complex number.
* **`n_dimensions = 4`**: Quaternion.
* **`n_dimensions = 8`**: Octonion.

## Specialized Agents

* **`Agent`**: Standard agent used by vector-based spaces (`SearchSpace`, `BooleanSpace`, `GridSpace`, `HyperComplexSpace`).
* **`GraphAgent`**: Agent whose position is a `Graph` or a `Tree` instead of a numeric array, used by `TreeSpace` and `GraphSpace`.

## Example Code

```python
from opytimizer.core import Agent

# Number of decision variables, dimensions and objectives
n_variables = 2
n_dimensions = 1
n_objectives = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0, 0]
upper_bound = [1, 1]

# Creates a new Agent
a = Agent(n_variables, n_dimensions, n_objectives, lower_bound, upper_bound)

# Prints out some properties
print(a.position, a.fit)
print(a.mapped_position)
```
