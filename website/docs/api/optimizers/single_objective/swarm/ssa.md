---
title: SSA
description: API reference for SSA.
---

# `SSA`

**Module:** `opytimizer.optimizers.single_objective.swarm.ssa`

A SSA class, inherited from Optimizer.

This is the designed class to define SSA-related
variables and methods.

References:
    S. Mirjalili et al. Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
    Advances in Engineering Software (2017).

## Constructor

```python
SSA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Salp Swarm Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
