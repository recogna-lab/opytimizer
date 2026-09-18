---
title: AdaptationMechanism
description: API reference for AdaptationMechanism.
---

# `AdaptationMechanism`

**Module:** `opytimizer.hyperheuristics.adaptation_mechanism`

Abstract base class for adaptation mechanisms.

## Constructor

```python
AdaptationMechanism() -> None
```

## Methods

### `adapt`

```python
adapt(self, optimizer: Any, performance_history: List[float], iteration: int) -> Dict[str, Any]
```

Adapt optimizer parameters based on performance.

Returns:
    (Dict[str, Any]): Adapted parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |
| `performance_history` | `List[float]` |  | Performance history of the optimizer. |
| `iteration` | `int` |  | Current iteration number. |
