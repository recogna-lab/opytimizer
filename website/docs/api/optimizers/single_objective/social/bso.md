---
title: BSO
description: API reference for BSO.
---

# `BSO`

**Module:** `opytimizer.optimizers.single_objective.social.bso`

A BSO class, inherited from Optimizer.

This is the designed class to define BSO-related
variables and methods.

References:
    Y. Shi. Brain Storm Optimization Algorithm.
    International Conference in Swarm Intelligence (2011).

## Constructor

```python
BSO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Brain Storm Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Number of iterations.s |
