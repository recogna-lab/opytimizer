---
title: Environment
description: API reference for Environment.
---

# `Environment`

**Module:** `opytimizer.core.environment`

Wrapper API class for backend control logic

## Constructor

```python
Environment(device: "Literal['numpy', 'cupy']" = 'numpy', dtype: 'str' = 'float32')
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | `Literal['numpy', 'cupy']` | `'numpy'` | — |
| `dtype` | `str` | `'float32'` | — |

## Methods

### `reset`

```python
reset(self) -> 'Environment'
```

Resets to the standard values

### `set_backend`

```python
set_backend(self, backend: "Literal['numpy', 'cupy'] | Backend") -> 'Environment'
```

Defines computational backend

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | `Literal['numpy', 'cupy'] | Backend` |  | — |

### `set_dtype`

```python
set_dtype(self, dtype: 'str') -> 'Environment'
```

Defines arrays data type

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dtype` | `str` |  | — |
