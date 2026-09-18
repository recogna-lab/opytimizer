---
title: History
description: API reference for History.
---

# `History`

**Module:** `opytimizer.utils.history`

A History class is responsible for saving each iteration's output.

Note that you can use dump() and parse() for whatever your needs. Our default
is only for agents, best agent and best agent's index.

## Constructor

```python
History(save_agents: bool = False) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `save_agents` | `bool` | `False` | — |

## Methods

### `dump`

```python
dump(self, **kwargs) -> None
```

Dumps keyword pairs into self-class attributes.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `kwargs` |  |  | — |

### `get_convergence`

```python
get_convergence(self, key: str, index: Optional[Tuple[int, ...]] = 0) -> numpy.ndarray
```

Gets the convergence list of a specified key.

Returns:
    (np.ndarray): Values based on key and index.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `key` | `str` |  | Key to be retrieved. |
| `index` | `Optional[Tuple[int, ...]]` | `0` | Index to be retrieved. |
