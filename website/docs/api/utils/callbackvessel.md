---
title: CallbackVessel
description: API reference for CallbackVessel.
---

# `CallbackVessel`

**Module:** `opytimizer.utils.callback`

Wraps multiple callbacks in an ready-to-use class.

## Constructor

```python
CallbackVessel(callbacks: List[opytimizer.utils.callback.Callback]) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `callbacks` | `List[opytimizer.utils.callback.Callback]` |  | — |

## Methods

### `on_evaluate_after`

```python
on_evaluate_after(self, *evaluate_args) -> None
```

Performs a list of callbacks after the `evaluate` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluate_args` |  |  | — |

### `on_evaluate_before`

```python
on_evaluate_before(self, *evaluate_args) -> None
```

Performs a list of callbacks prior to the `evaluate` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluate_args` |  |  | — |

### `on_iteration_begin`

```python
on_iteration_begin(self, iteration: int, opt_model: ~Opytimizer) -> None
```

Performs a list of callbacks whenever an iteration begins.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `iteration` | `int` |  | Current iteration. |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_iteration_end`

```python
on_iteration_end(self, iteration: int, opt_model: ~Opytimizer) -> None
```

Performs a list of callbacks whenever an iteration ends.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `iteration` | `int` |  | Current iteration. |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_task_begin`

```python
on_task_begin(self, opt_model: ~Opytimizer) -> None
```

Performs a list of callbacks whenever a task begins.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_task_end`

```python
on_task_end(self, opt_model: ~Opytimizer) -> None
```

Performs a list of callbacks whenever a task ends.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt_model` | `~Opytimizer` |  | An instance of the optimization model. |

### `on_update_after`

```python
on_update_after(self, *update_args) -> None
```

Performs a list of callbacks after the `update` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `update_args` |  |  | — |

### `on_update_before`

```python
on_update_before(self, *update_args) -> None
```

Performs a list of callbacks prior to the `update` method.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `update_args` |  |  | — |
