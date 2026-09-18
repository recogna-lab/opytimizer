---
title: GWO
description: API reference for GWO.
---

# `GWO`

**Module:** `opytimizer.optimizers.single_objective.population.gwo`

A GWO class, inherited from Optimizer.

This is the designed class to define GWO-related
variables and methods.

References:
    S. Mirjalili, S. Mirjalili and A. Lewis. Grey Wolf Optimizer.
    Advances in Engineering Software (2014).

## Constructor

```python
GWO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Grey Wolf Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
