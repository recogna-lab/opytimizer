---
title: AIWPSO
description: API reference for AIWPSO.
---

# `AIWPSO`

**Module:** `opytimizer.optimizers.single_objective.swarm.pso`

An AIWPSO class, inherited from PSO.

This is the designed class to define AIWPSO-related
variables and methods.

References:
    A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
    A novel particle swarm optimization algorithm with adaptive inertia weight.
    Applied Soft Computing (2011).

## Constructor

```python
AIWPSO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int) -> None
```

Wraps Adaptive Inertia Weight Particle Swarm Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
