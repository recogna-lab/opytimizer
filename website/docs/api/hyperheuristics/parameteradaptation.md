---
title: ParameterAdaptation
description: API reference for ParameterAdaptation.
---

# `ParameterAdaptation`

**Module:** `opytimizer.hyperheuristics.adaptation_mechanism`

Parameter adaptation mechanism.

## Constructor

```python
ParameterAdaptation(adaptation_rate: float = 0.1) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `adaptation_rate` | `float` | `0.1` | — |

## Methods

### `adapt`

```python
adapt(self, optimizer: Any, performance_history: List[float], iteration: int) -> Dict[str, Any]
```

Adapt optimizer parameters.

Returns:
    (Dict[str, Any]): Adapted parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |
| `performance_history` | `List[float]` |  | Performance history of the optimizer. |
| `iteration` | `int` |  | Current iteration number. |
