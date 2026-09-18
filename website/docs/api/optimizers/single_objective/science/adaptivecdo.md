---
title: AdaptiveCDO
description: API reference for AdaptiveCDO.
---

# `AdaptiveCDO`

**Module:** `opytimizer.optimizers.single_objective.science.cdo`

Adaptive Chernobyl Disaster Optimizer.

This variant implements adaptive parameter control mechanisms.

## Constructor

```python
AdaptiveCDO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `adapt_parameters`

```python
adapt_parameters(self, success_rate: float) -> None
```

Adapt parameters based on success rate.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `success_rate` | `float` |  | Rate of successful updates |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Updates using adaptive parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information |
| `function` | `opytimizer.core.function.Function` |  | Objective function |
| `iteration` | `int` |  | Current iteration |
| `n_iterations` | `int` |  | Maximum iterations |
