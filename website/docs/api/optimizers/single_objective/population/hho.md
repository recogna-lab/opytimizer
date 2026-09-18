---
title: HHO
description: API reference for HHO.
---

# `HHO`

**Module:** `opytimizer.optimizers.single_objective.population.hho`

An HHO class, inherited from Optimizer.

This is the designed class to define HHO-related
variables and methods.

References:
    A. Heidari et al. Harris hawks optimization: Algorithm and applications.
    Future Generation Computer Systems (2019).

## Constructor

```python
HHO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Harris Hawks Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
