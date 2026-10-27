---
sidebar_position: 13
title: Hyper-Heuristics
---

# Hyper-Heuristics

Learn how to combine several low-level optimizers into a single high-level optimizer using selection, adaptive, and hybrid hyper-heuristics.

## Selection Hyper-Heuristic

A `SelectionHyperHeuristic` chooses, every `selection_interval` iterations, which optimizer updates the population.

```python
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.hyperheuristics.selection import SelectionHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import ChoiceFunction
from opytimizer.optimizers.single_objective.swarm import ABC, PSO
from opytimizer.spaces import SearchSpace


# Defines a performance metric function
# Note that it receives the list of agents and returns a value to be minimized
def performance_metric(agents):
    return min(agent.fit for agent in agents)


# One should declare a list of optimizer instances to be used by the hyperheuristic
optimizers = [
    PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}),
    ABC(params={"n_trials": 10}),
]

# Creates the space and function
space = SearchSpace(
    n_agents=20, n_variables=2, n_objectives=1, lower_bound=[-10, -10], upper_bound=[10, 10]
)
function = Function(Sphere())

# Creates a SelectionHyperHeuristic
h = SelectionHyperHeuristic(
    optimizers=optimizers,
    selection_strategy=ChoiceFunction(),
    performance_metric=performance_metric,
    selection_interval=10,
)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, h, function)

# Runs the hyperheuristic optimization
opt.start(MaxIterations(50))

# Prints out how many times each optimizer was selected
stats = h.get_strategy_statistics()
print(f"Optimizer selections: {stats['optimizer_selections']}")
```

## Adaptive Hyper-Heuristic

An `AdaptiveHyperHeuristic` adjusts the parameters of its optimizers every `adaptation_interval` iterations.

```python
from opytimizer.hyperheuristics.adaptation_mechanism import ParameterAdaptation
from opytimizer.hyperheuristics.adaptive import AdaptiveHyperHeuristic
from opytimizer.optimizers.single_objective.swarm import ABC, PSO

# Creates an AdaptiveHyperHeuristic
h = AdaptiveHyperHeuristic(
    optimizers=[PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}), ABC(params={"n_trials": 10})],
    adaptation_mechanism=ParameterAdaptation(adaptation_rate=0.1),
    adaptation_interval=20,
)
```

## Hybrid Hyper-Heuristic

A `HybridHyperHeuristic` combines both selection and adaptation.

```python
from opytimizer.hyperheuristics.adaptation_mechanism import StrategyAdaptation
from opytimizer.hyperheuristics.hybrid import HybridHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import MultiArmedBandit
from opytimizer.optimizers.single_objective.swarm import ABC, PSO

# Creates a HybridHyperHeuristic
h = HybridHyperHeuristic(
    optimizers=[PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}), ABC(params={"n_trials": 10})],
    selection_strategy=MultiArmedBandit(),
    adaptation_mechanism=StrategyAdaptation(),
    selection_interval=10,
    adaptation_interval=20,
)
```

Both hyper-heuristics are used exactly like the selection one: pass them as the optimizer of `Opytimizer`. Their statistics are available through `get_adaptation_statistics()` and `get_hybrid_statistics()`, respectively.

## Tracking Performance

The `PerformanceTrackingCallback` tracks the performance of each low-level optimizer within a sliding window:

```python
from opytimizer.utils.callback import PerformanceTrackingCallback

# Creates the callback and continues the optimization task until the 100th iteration
performance_callback = PerformanceTrackingCallback(window_size=5)
opt.start(MaxIterations(100), callbacks=[performance_callback])

# Gathers the statistics from the callback
stats = performance_callback.get_statistics()
print(f"Performance ranking: {stats['performance_ranking']}")
```
