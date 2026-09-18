---
title: SAVPSO
description: API reference for SAVPSO.
---

# `SAVPSO`

**Module:** `opytimizer.optimizers.single_objective.swarm.pso`

An SAVPSO class, inherited from Optimizer.

This is the designed class to define SAVPSO-related
variables and methods.

References:
    H. Lu and W. Chen.
    Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
    Journal of global optimization (2008).

## Constructor

```python
SAVPSO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Wraps Self-adaptive Velocity Particle Swarm Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information. |
