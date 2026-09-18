---
title: PopulationAdaptation
description: API reference for PopulationAdaptation.
---

# `PopulationAdaptation`

**Module:** `opytimizer.hyperheuristics.adaptation_mechanism`

Population adaptation mechanism.

## Constructor

```python
PopulationAdaptation(min_population: int = 10, max_population: int = 100) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `min_population` | `int` | `10` | — |
| `max_population` | `int` | `100` | — |

## Methods

### `adapt`

```python
adapt(self, optimizer: Any, performance_history: List[float], iteration: int) -> Dict[str, Any]
```

Adapt population size.

Returns:
    (Dict[str, Any]): Adapted population parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |
| `performance_history` | `List[float]` |  | Performance history of the optimizer. |
| `iteration` | `int` |  | Current iteration number. |
