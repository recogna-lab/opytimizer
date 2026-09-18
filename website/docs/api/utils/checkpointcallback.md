---
title: CheckpointCallback
description: API reference for CheckpointCallback.
---

# `CheckpointCallback`

**Module:** `opytimizer.utils.callback`

A CheckpointCallback class that handles additional logging and
model's checkpointing.

## Constructor

```python
CheckpointCallback(file_path: str = None, frequency: int = 0) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `file_path` | `str` | `None` | — |
| `frequency` | `int` | `0` | — |

## Methods

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
