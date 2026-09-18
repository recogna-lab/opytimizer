---
title: IHS
description: API reference for IHS.
---

# `IHS`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.hs`

An IHS class, inherited from HS.

This is the designed class to define IHS-related
variables and methods.

References:
    M. Mahdavi, M. Fesanghary, and E. Damangir.
    An improved harmony search algorithm for solving optimization problems.
    Applied Mathematics and Computation (2007).

## Constructor

```python
IHS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Improved Harmony Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
