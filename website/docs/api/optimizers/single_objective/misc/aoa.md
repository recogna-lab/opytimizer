---
title: AOA
description: API reference for AOA.
---

# `AOA`

**Module:** `opytimizer.optimizers.single_objective.misc.aoa`

An AOA class, inherited from Optimizer.

This is the designed class to define AOA-related
variables and methods.

References:
    L. Abualigah et al. The Arithmetic Optimization Algorithm.
    Computer Methods in Applied Mechanics and Engineering (2021).

## Constructor

```python
AOA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Arithmetic Optimization Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
