---
title: TWO
description: API reference for TWO.
---

# `TWO`

**Module:** `opytimizer.optimizers.single_objective.science.two`

A TWO class, inherited from Optimizer.

This is the designed class to define TWO-related
variables and methods.

References:
    A. Kaveh. Tug of War Optimization.
    Advances in Metaheuristic Algorithms for Optimal Design of Structures (2016).

## Constructor

```python
TWO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Tug of War Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
