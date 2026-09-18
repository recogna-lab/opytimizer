---
title: HyperHeuristic
description: API reference for HyperHeuristic.
---

# `HyperHeuristic`

**Module:** `opytimizer.core.hyperheuristic`

A HyperHeuristic class that manages multiple low-level optimizers
and provides high-level strategies for algorithm selection and adaptation.

It supports both single-objective and multi-objective optimization by allowing
a custom performance_metric function to be passed (e.g., min_fitness, hypervolume, etc).

It also supports parameter adaptation and strategy adaptation mechanisms.

## Constructor

```python
HyperHeuristic(optimizers: Optional[List[opytimizer.core.optimizer.Optimizer]] = None, performance_metric: Optional[Callable[[Any], float]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `Optional[List[opytimizer.core.optimizer.Optimizer]]` | `None` | — |
| `performance_metric` | `Optional[Callable[[Any], float]]` | `None` | — |

## Methods

### `add_optimizer`

```python
add_optimizer(self, optimizer: opytimizer.core.optimizer.Optimizer) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `opytimizer.core.optimizer.Optimizer` |  | — |

### `compile`

```python
compile(self, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]) -> None
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]` |  | — |

### `evaluate`

```python
evaluate(self, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace], function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

If you need a specific evaluate method, please re-implement
it on child's class.

Also, note that function only accept arguments that are
found on Opytimizer class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `get_average_performance`

```python
get_average_performance(self, optimizer_name: str) -> Optional[float]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer_name` | `str` |  | — |

### `get_best_performance`

```python
get_best_performance(self, optimizer_name: str) -> Optional[float]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer_name` | `str` |  | — |

### `get_statistics`

```python
get_statistics(self) -> Dict[str, Any]
```

### `remove_optimizer`

```python
remove_optimizer(self, optimizer_name: str) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer_name` | `str` |  | — |

### `select_optimizer`

```python
select_optimizer(self, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace], function: opytimizer.core.function.Function) -> opytimizer.core.optimizer.Optimizer
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |

### `update`

```python
update(self, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace], function: opytimizer.core.function.Function = None) -> None
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]` |  | — |
| `function` | `opytimizer.core.function.Function` | `None` | — |

### `update_performance`

```python
update_performance(self, optimizer: opytimizer.core.optimizer.Optimizer, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `opytimizer.core.optimizer.Optimizer` |  | — |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]` |  | — |
