---
title: StrategyAdaptation
description: API reference for StrategyAdaptation.
---

# `StrategyAdaptation`

**Module:** `opytimizer.hyperheuristics.adaptation_mechanism`

Strategy adaptation mechanism.

## Constructor

```python
StrategyAdaptation(adaptation_threshold: float = 0.1) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `adaptation_threshold` | `float` | `0.1` | — |

## Methods

### `adapt`

```python
adapt(self, optimizer: Any, performance_history: List[float], iteration: int) -> Dict[str, Any]
```

Adapt optimization strategy.

Returns:
    (Dict[str, Any]): Adapted strategy parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |
| `performance_history` | `List[float]` |  | Performance history of the optimizer. |
| `iteration` | `int` |  | Current iteration number. |
