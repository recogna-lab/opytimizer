---
title: NoImprovement
description: API reference for NoImprovement.
---

# `NoImprovement`

**Module:** `opytimizer.core.stopping`

`patience` non-improvement iterations.

When the best fitness lives on a CUDA device (tensorized spaces), comparing
it against a Python float forces a device -> host synchronization on every
call, which drains the whole kernel queue. To avoid that, the running best
and the non-improvement counter are kept ON the device and updated with
async kernels; the counter is only read back every `check_every` calls.

## Constructor

```python
NoImprovement(patience: int, min_delta: float = 1e-08, check_every: int = 1) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `patience` | `int` |  | Number of consecutive non-improving iterations tolerated. |
| `min_delta` | `float` | `1e-08` | Minimum decrease that counts as an improvement. |
| `check_every` | `int` | `1` | Read the counter back to the host every this many calls. The default (1) keeps the exact semantics of the original implementation. On GPU, a larger value (e.g. 10) removes most of the synchronizations at the price of overshooting the stop by at most `check_every - 1` iterations. It has no effect on the cost of a host-resident (numpy) run. |

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
