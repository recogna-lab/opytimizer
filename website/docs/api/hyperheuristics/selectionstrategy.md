---
title: SelectionStrategy
description: API reference for SelectionStrategy.
---

# `SelectionStrategy`

**Module:** `opytimizer.hyperheuristics.selection_strategy`

Abstract base class for selection strategies.

## Constructor

```python
SelectionStrategy() -> None
```

## Methods

### `select`

```python
select(self, optimizers: List[Any], performance_history: Dict[str, List[float]]) -> Any
```

Select an optimizer based on the strategy.

Returns:
    (Any): Selected optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `List[Any]` |  | List of available optimizers. |
| `performance_history` | `Dict[str, List[float]]` |  | Performance history for each optimizer. |
