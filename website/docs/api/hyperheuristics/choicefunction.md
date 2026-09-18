---
title: ChoiceFunction
description: API reference for ChoiceFunction.
---

# `ChoiceFunction`

**Module:** `opytimizer.hyperheuristics.selection_strategy`

Choice Function selection strategy.

## Constructor

```python
ChoiceFunction(alpha: float = 0.5, beta: float = 0.5) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.5` | — |
| `beta` | `float` | `0.5` | — |

## Methods

### `select`

```python
select(self, optimizers: List[Any], performance_history: Dict[str, List[float]]) -> Any
```

Select optimizer using Choice Function.

Returns:
    (Any): Selected optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `List[Any]` |  | List of available optimizers. |
| `performance_history` | `Dict[str, List[float]]` |  | Performance history for each optimizer. |
