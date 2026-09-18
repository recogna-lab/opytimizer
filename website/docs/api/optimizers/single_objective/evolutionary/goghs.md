---
title: GOGHS
description: API reference for GOGHS.
---

# `GOGHS`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.hs`

A GOGHS class, inherited from NGHS.

This is the designed class to define GOGHS-related
variables and methods.

References:
    Z. Guo, S. Wang, X. Yue and H. Yang.
    Global harmony search with generalized opposition-based learning.
    Soft Computing (2017).

## Constructor

```python
GOGHS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Generalized Opposition Global-Best Harmony Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
