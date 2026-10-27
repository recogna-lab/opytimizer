---
sidebar_position: 18
title: Visualization
---

# Visualization

Learn how to plot convergence curves, Pareto fronts, and population distributions using the `opytimizer.visualization` package.

## Available Plots

* **`plot_convergence`**: Compares the best fitness of single-objective optimizers over the iterations.
* **`plot_agents`**: Plots a set of agents (e.g., the Pareto front) in a 2D or 3D objective space.
* **`plot_pareto_front_evolution`**: Plots the Pareto front at specific iterations.
* **`plot_pareto_front_comparison`**: Compares the final Pareto fronts of multiple optimizers.
* **`plot_population_distribution_histogram`**: Plots the distribution of an objective over the population.
* **`plot_graph`**: Plots a `Tree` or a `Graph`, such as the best Genetic Programming solution.

Every plot accepts a `backend` argument (`"matplotlib"` or `"plotly"`) and returns an object with `show()` and `save(path)` methods.

## Convergence Plot

```python
import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.evolutionary import DE
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.visualization import plot_convergence

# Number of agents and decision variables
n_agents = 20
n_variables = 2
n_objectives = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

function = Function(Sphere())

# Runs each optimizer over the same problem
# Note that `save_history` is required to gather the convergence
histories = []
for optimizer in [DE(), PSO()]:
    np.random.seed(0)

    space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
    opt = Opytimizer(space, optimizer, function, save_history=True)
    opt.start(MaxIterations(250))

    histories.append(opt.history.best_agent)

# Plots the convergence of both optimizers
plot_convergence(*histories, labels=["DE", "PSO"], backend="matplotlib").show()
```

## Pareto Front Plots

```python
import numpy as np

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.spaces import SearchSpace
from opytimizer.utils.operators import PolynomialMutation, SBXCrossover
from opytimizer.visualization import plot_agents, plot_pareto_front_evolution


def zdt1(x: np.ndarray) -> np.ndarray:
    x = x.flatten()
    n = len(x)

    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1.0)
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h

    return np.array([f1, f2])


# Random seed for experimental consistency
np.random.seed(0)

# Number of agents, decision variables and objectives
n_agents = 100
n_variables = 30
n_objectives = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0.0] * n_variables
upper_bound = [1.0] * n_variables

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = NSGA2(
    mutation_operator=PolynomialMutation(eta=20, rate=(1.0 / n_variables)),
    crossover_operator=SBXCrossover(eta=30, rate=1.0, gene_rate=1.0, n_offspring=2),
)
function = Function(zdt1)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_history=True)

# Runs the optimization task
opt.start(MaxIterations(100))

# Plots the final Pareto front
plot_agents(
    opt.space.pareto_front, backend="matplotlib", title="NSGA2 - Pareto Front", labels=["F1", "F2"]
).show()

# Plots the Pareto front at specific iterations
plot_pareto_front_evolution(
    opt.history.pareto_front,
    backend="matplotlib",
    title="NSGA2 - Pareto Front Evolution",
    labels=["F1", "F2"],
    iterations=[1, 25, 50, 75, 100],
).show()
```
