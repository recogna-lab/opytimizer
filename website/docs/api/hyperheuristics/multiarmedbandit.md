---
title: MultiArmedBandit
description: API reference for MultiArmedBandit.
---

# `MultiArmedBandit`

**Module:** `opytimizer.hyperheuristics.selection_strategy`

Multi-Armed Bandit selection strategy using Upper Confidence Bound (UCB).

## Constructor

```python
MultiArmedBandit(exploration_constant: float = 2.0) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `exploration_constant` | `float` | `2.0` | — |

## Methods

### `select`

```python
select(self, optimizers: List[Any], performance_history: Dict[str, List[float]]) -> Any
```

Select optimizer using UCB (Upper Confidence Bound).

Returns:
    (Any): Selected optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `List[Any]` |  | List of available optimizers. |
| `performance_history` | `Dict[str, List[float]]` |  | Performance history for each optimizer. |
