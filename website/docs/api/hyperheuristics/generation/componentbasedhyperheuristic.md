---
title: ComponentBasedHyperHeuristic
description: API reference for ComponentBasedHyperHeuristic.
---

# `ComponentBasedHyperHeuristic`

**Module:** `opytimizer.hyperheuristics.generation.component_based`

A Component-Based Hyperheuristic that combines different
components from various optimization algorithms.

This hyperheuristic generates new algorithms by selecting and
combining different components (initialization, selection,
variation, etc.) from existing algorithms.

## Constructor

```python
ComponentBasedHyperHeuristic(components: Optional[Dict[str, List[Callable]]] = None, population_size: int = 30, crossover_rate: float = 0.8, mutation_rate: float = 0.2, performance_metric=None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `components` | `Optional[Dict[str, List[Callable]]]` | `None` | — |
| `population_size` | `int` | `30` | — |
| `crossover_rate` | `float` | `0.8` | — |
| `mutation_rate` | `float` | `0.2` | — |
| `performance_metric` |  | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._Space) -> None
```

Compile the component-based hyperheuristic.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._Space` |  | A Space object containing meta-information. |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._Space, function: opytimizer.core.function.Function) -> None
```

Evaluate the population of generated algorithms.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._Space` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `get_component_statistics`

```python
get_component_statistics(self) -> Dict[str, Any]
```

Get statistics about the component-based evolution.

Returns:
    (Dict[str, Any]): Dictionary containing component statistics.

### `update`

```python
update(self, space: opytimizer.core.space._Space) -> None
```

Update the population through evolution.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._Space` |  | A Space object containing agents and update-related information. |
