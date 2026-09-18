---
title: RandomDescent
description: API reference for RandomDescent.
---

# `RandomDescent`

**Module:** `opytimizer.hyperheuristics.selection_strategy`

Random Descent selection strategy.

## Constructor

```python
RandomDescent(acceptance_threshold: float = 0.1) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `acceptance_threshold` | `float` | `0.1` | — |

## Methods

### `select`

```python
select(self, optimizers: List[Any], performance_history: Dict[str, List[float]]) -> Any
```

Select optimizer using Random Descent.

Returns:
    (Any): Selected optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `List[Any]` |  | List of available optimizers. |
| `performance_history` | `Dict[str, List[float]]` |  | Performance history for each optimizer. |
