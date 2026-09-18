---
title: HC
description: API reference for HC.
---

# `HC`

**Module:** `opytimizer.optimizers.single_objective.misc.hc`

An HC class, inherited from Optimizer.

This is the designed class to define HC-related
variables and methods.

References:
    S. Skiena. The Algorithm Design Manual (2010).

## Constructor

```python
HC(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Hill Climbing over all agents and variables (p. 252).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
