---
title: BWO
description: API reference for BWO.
---

# `BWO`

**Module:** `opytimizer.optimizers.single_objective.swarm.bwo`

A BWO class, inherited from Optimizer.

This is the designed class to define BWO-related
variables and methods.

References:
    V. Hayyolalam and A. Kazem.
    Black Widow Optimization Algorithm: A novel meta-heuristic approach
    for solving engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2020).

## Constructor

```python
BWO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Wraps Black Widow Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
