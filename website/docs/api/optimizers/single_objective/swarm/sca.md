---
title: SCA
description: API reference for SCA.
---

# `SCA`

**Module:** `opytimizer.optimizers.single_objective.swarm.sca`

A SCA class, inherited from Optimizer.

This is the designed class to define SCA-related
variables and methods.

References:
    S. Mirjalili. SCA: A Sine Cosine Algorithm for solving optimization problems.
    Knowledge-Based Systems (2016).

## Constructor

```python
SCA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Sine Cosine Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
