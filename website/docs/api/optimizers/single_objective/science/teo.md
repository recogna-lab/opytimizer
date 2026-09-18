---
title: TEO
description: API reference for TEO.
---

# `TEO`

**Module:** `opytimizer.optimizers.single_objective.science.teo`

A TEO class, inherited from Optimizer.

This is the designed class to define TEO-related
variables and methods.

References:
    A. Kaveh and A. Dadras. A novel meta-heuristic optimization algorithm: Thermal exchange optimization.
    Advances in Engineering Software (2017).

## Constructor

```python
TEO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object containing meta-information. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int, n_iterations: int) -> None
```

Wraps Thermal Exchange Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
