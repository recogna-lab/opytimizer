---
title: WEO
description: API reference for WEO.
---

# `WEO`

**Module:** `opytimizer.optimizers.single_objective.science.weo`

A WEO class, inherited from Optimizer.

This is the designed class to define WEO-related
variables and methods.

References:
    A. Kaveh and T. Bakhshpoori.
    Water Evaporation Optimization: A novel physically inspired optimization algorithm.
    Computers & Structures (2016).

## Constructor

```python
WEO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Water Evaporation Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
