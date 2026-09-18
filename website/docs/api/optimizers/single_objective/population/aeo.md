---
title: AEO
description: API reference for AEO.
---

# `AEO`

**Module:** `opytimizer.optimizers.single_objective.population.aeo`

An AEO class, inherited from Optimizer.

This is the designed class to define AEO-related
variables and methods.

References:
    W. Zhao, L. Wang and Z. Zhang.
    Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm.
    Neural Computing and Applications (2019).

## Constructor

```python
AEO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Artificial Ecosystem-based Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
