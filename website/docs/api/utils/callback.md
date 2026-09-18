---
title: Callback
description: API reference for Callback.
---

# `Callback`

**Module:** `opytimizer.utils.callback`

A Callback class that handles additional variables and methods
manipulation that are not provided by the library.

## Constructor

```python
Callback()
```

## Methods

### `on_evaluate_after`

```python
on_evaluate_after(self, *evaluate_args) -> None
```

Performs a callback after the `evaluate` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluate_args` |  |  | — |

### `on_evaluate_before`

```python
on_evaluate_before(self, *evaluate_args) -> None
```

Performs a callback prior to the `evaluate` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluate_args` |  |  | — |

### `on_iteration_begin`

```python
on_iteration_begin(self, iteration: int, opt_model: ~Opytimizer) -> None
```

Performs a callback whenever an iteration begins.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `iteration` | `int` |  | Current iteration. |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_iteration_end`

```python
on_iteration_end(self, iteration: int, opt_model: ~Opytimizer) -> None
```

Performs a callback whenever an iteration ends.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `iteration` | `int` |  | Current iteration. |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_task_begin`

```python
on_task_begin(self, opt_model: ~Opytimizer) -> None
```

Performs a callback whenever a task begins.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_task_end`

```python
on_task_end(self, opt_model: ~Opytimizer) -> None
```

Performs a callback whenever a task ends.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_update_after`

```python
on_update_after(self, *update_args) -> None
```

Performs a callback after the `update` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `update_args` |  |  | — |

### `on_update_before`

```python
on_update_before(self, *update_args) -> None
```

Performs a callback prior to the `update` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `update_args` |  |  | — |
