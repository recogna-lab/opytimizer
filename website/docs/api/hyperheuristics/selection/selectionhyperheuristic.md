---
title: SelectionHyperHeuristic
description: API reference for SelectionHyperHeuristic.
---

# `SelectionHyperHeuristic`

**Module:** `opytimizer.hyperheuristics.selection.selection_hyperheuristic`

A selection-based hyperheuristic that uses different strategies
to select between low-level optimizers.

This hyperheuristic implements the selection approach where different
low-level optimizers are selected based on their performance history
and the chosen selection strategy.

## Constructor

```python
SelectionHyperHeuristic(optimizers: Optional[List[Any]] = None, selection_strategy: Optional[opytimizer.hyperheuristics.selection_strategy.SelectionStrategy] = None, performance_metric=None, selection_interval: int = 1) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `Optional[List[Any]]` | `None` | — |
| `selection_strategy` | `Optional[opytimizer.hyperheuristics.selection_strategy.SelectionStrategy]` | `None` | — |
| `performance_metric` |  | `None` | — |
| `selection_interval` | `int` | `1` | — |

## Methods

### `get_strategy_statistics`

```python
get_strategy_statistics(self) -> Dict[str, Any]
```

Get statistics about the selection strategy.

Returns:
    (Dict[str, Any]): Dictionary containing strategy statistics.

### `select_optimizer`

```python
select_optimizer(self, space: opytimizer.core.space._Space, function: opytimizer.core.function.Function) -> Any
```

Select optimizer using the configured selection strategy.

Returns:
    (Any): Selected optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._Space` |  | Current search space. |
| `function` | `opytimizer.core.function.Function` |  | Objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._Space, function: opytimizer.core.function.Function = None) -> None
```

Update the search space and potentially select a new optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._Space` |  | A Space object containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` | `None` | Objective function (optional, for optimizers that need it). |
