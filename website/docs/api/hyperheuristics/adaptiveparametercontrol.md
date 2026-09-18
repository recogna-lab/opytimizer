---
title: AdaptiveParameterControl
description: API reference for AdaptiveParameterControl.
---

# `AdaptiveParameterControl`

**Module:** `opytimizer.hyperheuristics.adaptation_mechanism`

Adaptive parameter control mechanism.

## Constructor

```python
AdaptiveParameterControl(adaptation_window: int = 10) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `adaptation_window` | `int` | `10` | — |

## Methods

### `adapt`

```python
adapt(self, optimizer: Any, performance_history: List[float], iteration: int) -> Dict[str, Any]
```

Adapt parameters using adaptive control.

Returns:
    (Dict[str, Any]): Adapted parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |
| `performance_history` | `List[float]` |  | Performance history of the optimizer. |
| `iteration` | `int` |  | Current iteration number. |
