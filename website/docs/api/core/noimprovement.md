---
title: NoImprovement
description: API reference for NoImprovement.
---

# `NoImprovement`

**Module:** `opytimizer.core.stopping`

`patience` non-improvement iterations

## Constructor

```python
NoImprovement(patience: int, min_delta: float = 1e-08) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `patience` | `int` |  | — |
| `min_delta` | `float` | `1e-08` | — |

## Methods

### `init_pbar`

```python
init_pbar(self, position) -> None
```

Progress Bar initialization.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` |  |  | — |

### `reset`

```python
reset(self) -> None
```

Internal State reinicialization.

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
