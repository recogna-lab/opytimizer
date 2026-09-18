---
title: UMDA
description: API reference for UMDA.
---

# `UMDA`

**Module:** `opytimizer.optimizers.single_objective.boolean.umda`

An UMDA class, inherited from Optimizer.

This is the designed class to define UMDA-related variables and methods.

References:
    H. Mühlenbein. The equation for response to selection and its use for prediction.
    Evolutionary Computation (1997).

## Constructor

```python
UMDA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Wraps Univariate Marginal Distribution Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
