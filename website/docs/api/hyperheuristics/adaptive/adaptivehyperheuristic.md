---
title: AdaptiveHyperHeuristic
description: API reference for AdaptiveHyperHeuristic.
---

# `AdaptiveHyperHeuristic`

**Module:** `opytimizer.hyperheuristics.adaptive.adaptive_hyperheuristic`

An adaptive hyperheuristic that adapts optimizer parameters
based on performance feedback.

This hyperheuristic implements the adaptive approach where optimizer
parameters are modified dynamically based on their performance history
and the chosen adaptation mechanism.

## Constructor

```python
AdaptiveHyperHeuristic(optimizers: Optional[List[Any]] = None, adaptation_mechanism: Optional[opytimizer.hyperheuristics.adaptation_mechanism.AdaptationMechanism] = None, performance_metric=None, adaptation_interval: int = 5) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `Optional[List[Any]]` | `None` | — |
| `adaptation_mechanism` | `Optional[opytimizer.hyperheuristics.adaptation_mechanism.AdaptationMechanism]` | `None` | — |
| `performance_metric` |  | `None` | — |
| `adaptation_interval` | `int` | `5` | — |

## Methods

### `adapt_optimizer`

```python
adapt_optimizer(self, optimizer: Any) -> None
```

Adapt the parameters of an optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |

### `get_adaptation_statistics`

```python
get_adaptation_statistics(self) -> Dict[str, Any]
```

Get statistics about the adaptation mechanism.

Returns:
    (Dict[str, Any]): Dictionary containing adaptation statistics.

### `update`

```python
update(self, space: opytimizer.core.space._Space, function: opytimizer.core.function.Function = None) -> None
```

Update the search space and potentially adapt optimizer parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._Space` |  | A Space object containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` | `None` | Objective function (optional, for optimizers that need it). |
