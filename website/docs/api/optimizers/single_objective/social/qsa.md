---
title: QSA
description: API reference for QSA.
---

# `QSA`

**Module:** `opytimizer.optimizers.single_objective.social.qsa`

A QSA class, inherited from Optimizer.

This is the designed class to define QSA-related
variables and methods.

References:
    J. Zhang et al. Queuing search algorithm: A novel metaheuristic algorithm
    for solving engineering optimization problems.
    Applied Mathematical Modelling (2018).

## Constructor

```python
QSA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Wraps Queue Search Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
