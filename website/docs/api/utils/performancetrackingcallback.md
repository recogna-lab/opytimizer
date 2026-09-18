---
title: PerformanceTrackingCallback
description: API reference for PerformanceTrackingCallback.
---

# `PerformanceTrackingCallback`

**Module:** `opytimizer.utils.callback`

A PerformanceTrackingCallback class that tracks optimizer performance
during hyperheuristic optimization.

## Constructor

```python
PerformanceTrackingCallback(window_size: int = 10) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `window_size` | `int` | `10` | — |

## Methods

### `get_average_performance`

```python
get_average_performance(self, optimizer_name: str) -> Optional[float]
```

Get the average performance of an optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer_name` | `str` |  | — |

### `get_best_performance`

```python
get_best_performance(self, optimizer_name: str) -> Optional[float]
```

Get the best performance achieved by an optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer_name` | `str` |  | — |

### `get_performance_ranking`

```python
get_performance_ranking(self) -> List[tuple]
```

Get performance ranking of all optimizers.

### `get_selection_frequency`

```python
get_selection_frequency(self, optimizer_name: str) -> float
```

Get the selection frequency of an optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer_name` | `str` |  | — |

### `get_statistics`

```python
get_statistics(self) -> Dict[str, Any]
```

Get comprehensive statistics about all optimizers.

### `on_evaluate_after`

```python
on_evaluate_after(self, *evaluate_args) -> None
```

Track performance after evaluation.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluate_args` |  |  | — |

### `on_iteration_begin`

```python
on_iteration_begin(self, iteration: int, opt_model: ~Opytimizer) -> None
```

Track iteration start.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `iteration` | `int` |  | — |
| `opt_model` | `~Opytimizer` |  | — |

### `on_task_begin`

```python
on_task_begin(self, opt_model: ~Opytimizer) -> None
```

Initialize tracking when optimization begins.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt_model` | `~Opytimizer` |  | — |

### `on_update_after`

```python
on_update_after(self, *update_args) -> None
```

Track optimizer selection after update.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `update_args` |  |  | — |
