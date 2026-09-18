---
title: DiscreteSearchCallback
description: API reference for DiscreteSearchCallback.
---

# `DiscreteSearchCallback`

**Module:** `opytimizer.utils.callback`

A DiscreteSearchCallback class that handles mapping floating-point variables
to discrete values.

## Constructor

```python
DiscreteSearchCallback(allowed_values: List[Union[int, float]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `allowed_values` | `List[Union[int, float]]` | `None` | — |

## Methods

### `on_evaluate_before`

```python
on_evaluate_before(self, *evaluate_args) -> None
```

Performs a callback prior to the `evaluate` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluate_args` |  |  | — |

### `on_task_begin`

```python
on_task_begin(self, opt_model: ~Opytimizer) -> None
```

Performs a callback whenever a task begins.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |
