---
title: DE
description: API reference for DE.
---

# `DE`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.de`

A DE class, inherited from Optimizer.

This is the designed class to define DE-related
variables and methods.

References:
    R. Storn. On the usage of differential evolution for function optimization.
    Proceedings of North American Fuzzy Information Processing (1996).

## Constructor

```python
DE(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Differential Evolution over all agents and variables (eq. 1-4).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
