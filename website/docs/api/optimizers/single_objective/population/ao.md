---
title: AO
description: API reference for AO.
---

# `AO`

**Module:** `opytimizer.optimizers.single_objective.population.ao`

An AO class, inherited from Optimizer.

This is the designed class to define AO-related
variables and methods.

References:
    L. Abualigah et al. Aquila Optimizer: A novel meta-heuristic optimization Algorithm.
    Computers & Industrial Engineering (2021).

## Constructor

```python
AO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Aquila Optimizer over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
