---
title: MVO
description: API reference for MVO.
---

# `MVO`

**Module:** `opytimizer.optimizers.single_objective.science.mvo`

A MVO class, inherited from Optimizer.

This is the designed class to define MVO-related
variables and methods.

References:
    S. Mirjalili, S. M. Mirjalili and A. Hatamlou.
    Multi-verse optimizer: a nature-inspired algorithm for global optimization.
    Neural Computing and Applications (2016).

## Constructor

```python
MVO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Multi-Verse Optimizer over all agents and variables (eq. 3.1-3.4).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
