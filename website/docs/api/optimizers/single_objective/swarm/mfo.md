---
title: MFO
description: API reference for MFO.
---

# `MFO`

**Module:** `opytimizer.optimizers.single_objective.swarm.mfo`

A MFO class, inherited from Optimizer.

This is the designed class to define MFO-related
variables and methods.

References:
    S. Mirjalili. Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
    Knowledge-Based Systems (2015).

## Constructor

```python
MFO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int, n_iterations: int) -> None
```

Wraps Moth-Flame Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
