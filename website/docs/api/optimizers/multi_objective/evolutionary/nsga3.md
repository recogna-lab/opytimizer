---
title: NSGA3
description: API reference for NSGA3.
---

# `NSGA3`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.nsga3`

NSGA3 class, inherited from MultiObjectiveOptimizer.

Replaces the crowding distance operator of NSGA-II with a reference-point-based
niching strategy, making it effective for problems with four or more objectives.

References:
    K. Deb and H. Jain. An Evolutionary Many-Objective Optimization Algorithm
    Using Reference-Point-Based Nondominated Sorting Approach, Part I.
    IEEE Transactions on Evolutionary Computation (2014).

## Constructor

```python
NSGA3(params: dict = None, crossover_operator=None, mutation_operator=None, reference_points: numpy.ndarray = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `dict` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |
| `reference_points` | `numpy.ndarray` | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._MultiObjectiveSpace) -> None
```

Compiles additional information used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | A Space object containing meta-information. |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function)
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Wraps NSGA-III over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | Objective function used to evaluate offspring. |
