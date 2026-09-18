---
title: MaxEvaluations
description: API reference for MaxEvaluations.
---

# `MaxEvaluations`

**Module:** `opytimizer.core.stopping`

Soft check: verified at end of each iteration.
For exact enforcement, use Function(budget=n) instead.

## Constructor

```python
MaxEvaluations(n: int) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n` | `int` |  | — |

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
