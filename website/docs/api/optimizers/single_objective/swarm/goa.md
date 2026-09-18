---
title: GOA
description: API reference for GOA.
---

# `GOA`

**Module:** `opytimizer.optimizers.single_objective.swarm.goa`

A GOA class, inherited from Optimizer.

This is the designed class to define GOA-related
variables and methods.

References:
    S. Saremi, S. Mirjalili and A. Lewis. Grasshopper Optimisation Algorithm: Theory and application.
    Advances in Engineering Software (2017).

## Constructor

```python
GOA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Grasshopper Optimization Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
