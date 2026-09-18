---
title: SMA
description: API reference for SMA.
---

# `SMA`

**Module:** `opytimizer.optimizers.single_objective.science.sma`

A SMA class, inherited from Optimizer.

This is the designed class to define SMA-related
variables and methods.

References:
    S. Li, H. Chen, M. Wang, A. A. Heidari, S. Mirjalili
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems (2020).

## Constructor

```python
SMA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object containing meta-information. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int, n_iterations: int) -> None
```

Wraps Slime Mould Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | — |
| `n_iterations` | `int` |  | — |
