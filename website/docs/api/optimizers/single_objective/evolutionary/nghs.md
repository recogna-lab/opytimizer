---
title: NGHS
description: API reference for NGHS.
---

# `NGHS`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.hs`

A NGHS class, inherited from HS.

This is the designed class to define NGHS-related
variables and methods.

References:
    D. Zou, L. Gao, J. Wu and S. Li.
    Novel global harmony search algorithm for unconstrained problems.
    Neurocomputing (2010).

## Constructor

```python
NGHS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Novel Global Harmony Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
