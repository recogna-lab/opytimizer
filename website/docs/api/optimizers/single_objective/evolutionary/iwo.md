---
title: IWO
description: API reference for IWO.
---

# `IWO`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.iwo`

An IWO class, inherited from Optimizer.

This is the designed class to define IWO-related
variables and methods.

References:
    A. R. Mehrabian and C. Lucas. A novel numerical optimization algorithm inspired from weed colonization.
    Ecological informatics (2006).

## Constructor

```python
IWO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Invasive Weed Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
