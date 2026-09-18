---
title: GreedySelection
description: API reference for GreedySelection.
---

# `GreedySelection`

**Module:** `opytimizer.hyperheuristics.selection_strategy`

Greedy selection strategy - always selects the best performing optimizer.

## Constructor

```python
GreedySelection() -> None
```

## Methods

### `select`

```python
select(self, optimizers: List[Any], performance_history: Dict[str, List[float]]) -> Any
```

Select optimizer using Greedy strategy.

Returns:
    (Any): Selected optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `List[Any]` |  | List of available optimizers. |
| `performance_history` | `Dict[str, List[float]]` |  | Performance history for each optimizer. |
