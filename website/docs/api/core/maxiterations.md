---
title: MaxIterations
description: API reference for MaxIterations.
---

# `MaxIterations`

**Module:** `opytimizer.core.stopping`

Base Class.  `should_stop` must be implemented in each subclass

## Constructor

```python
MaxIterations(n: int) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n` | `int` |  | — |

## Methods

### `init_pbar`

```python
init_pbar(self, position: int) -> None
```

Progress Bar initialization.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `int` |  | — |

### `should_stop`

```python
should_stop(self, opt) -> bool
```

Return True when the criterio was satisfied

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt` |  |  | — |

### `update_pbar`

```python
update_pbar(self, opt) -> None
```

Update the Progress Bar to the current optimization state.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `opt` |  |  | — |
