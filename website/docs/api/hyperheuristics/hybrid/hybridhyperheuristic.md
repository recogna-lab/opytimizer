---
title: HybridHyperHeuristic
description: API reference for HybridHyperHeuristic.
---

# `HybridHyperHeuristic`

**Module:** `opytimizer.hyperheuristics.hybrid.hybrid_hyperheuristic`

A hybrid hyperheuristic that combines selection and adaptation approaches.

This hyperheuristic implements a hybrid approach where both optimizer
selection and parameter adaptation are used together to improve performance.

## Constructor

```python
HybridHyperHeuristic(optimizers: Optional[List[Any]] = None, selection_strategy: Optional[opytimizer.hyperheuristics.selection_strategy.SelectionStrategy] = None, adaptation_mechanism: Optional[opytimizer.hyperheuristics.adaptation_mechanism.AdaptationMechanism] = None, performance_metric=None, selection_interval: int = 1, adaptation_interval: int = 5) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `Optional[List[Any]]` | `None` | List of low-level optimizers to be managed. |
| `selection_strategy` | `Optional[opytimizer.hyperheuristics.selection_strategy.SelectionStrategy]` | `None` | Strategy for selecting optimizers. |
| `adaptation_mechanism` | `Optional[opytimizer.hyperheuristics.adaptation_mechanism.AdaptationMechanism]` | `None` | Mechanism for adapting optimizer parameters. |
| `performance_metric` |  | `None` | Function to evaluate optimizer performance. |
| `selection_interval` | `int` | `1` | Interval between optimizer selections. |
| `adaptation_interval` | `int` | `5` | Interval between parameter adaptations. |

## Methods

### `adapt_optimizer`

```python
adapt_optimizer(self, optimizer: Any) -> None
```

Adapt the parameters of an optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `Any` |  | Optimizer to be adapted. |

### `get_hybrid_statistics`

```python
get_hybrid_statistics(self) -> Dict[str, Any]
```

Get comprehensive statistics about the hybrid hyperheuristic.

Returns:
    (Dict[str, Any]): Dictionary containing hybrid statistics.

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

Update the search space, potentially select a new optimizer, and adapt parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._Space` |  | A Space object containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` | `None` | Objective function (optional, for optimizers that need it). |
