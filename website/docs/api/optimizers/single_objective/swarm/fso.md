---
title: FSO
description: API reference for FSO.
---

# `FSO`

**Module:** `opytimizer.optimizers.single_objective.swarm.fso`

A FSO class, inherited from Optimizer.

This is the designed class to define FSO-related
variables and methods.

References:
    G. Azizyan et al.
    Flying Squirrel Optimizer (FSO): A novel SI-based optimization algorithm for engineering problems.
    Iranian Journal of Optimization (2019).

## Constructor

```python
FSO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Flying Squirrel Optimizer over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
