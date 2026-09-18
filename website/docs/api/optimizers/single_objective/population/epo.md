---
title: EPO
description: API reference for EPO.
---

# `EPO`

**Module:** `opytimizer.optimizers.single_objective.population.epo`

An EPO class, inherited from Optimizer.

This is the designed class to define EPO-related
variables and methods.

References:
    G. Dhiman and V. Kumar. Emperor penguin optimizer: A bio-inspired algorithm for engineering problems.
    Knowledge-Based Systems (2018).

## Constructor

```python
EPO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Emperor Penguin Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
