---
title: WOA
description: API reference for WOA.
---

# `WOA`

**Module:** `opytimizer.optimizers.single_objective.swarm.woa`

A WOA class, inherited from Optimizer.

This is the designed class to define WOA-related
variables and methods.

References:
    S. Mirjalli and A. Lewis. The Whale Optimization Algorithm.
    Advances in Engineering Software (2016).

## Constructor

```python
WOA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int, n_iterations: int) -> None
```

Wraps Whale Optimization Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations |
