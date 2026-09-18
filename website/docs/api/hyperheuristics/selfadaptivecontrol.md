---
title: SelfAdaptiveControl
description: API reference for SelfAdaptiveControl.
---

# `SelfAdaptiveControl`

**Module:** `opytimizer.hyperheuristics.adaptation_mechanism`

Self-adaptive parameter control mechanism.

## Constructor

```python
SelfAdaptiveControl(learning_rate: float = 0.01) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `learning_rate` | `float` | `0.01` | — |

## Methods

### `adapt`

```python
adapt(self, optimizer: Any, performance_history: List[float], iteration: int) -> Dict[str, Any]
```

Adapt parameters using self-adaptive control.

Returns:
    (Dict[str, Any]): Adapted parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |
| `performance_history` | `List[float]` |  | Performance history of the optimizer. |
| `iteration` | `int` |  | Current iteration number. |
