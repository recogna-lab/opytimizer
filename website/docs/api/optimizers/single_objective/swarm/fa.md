---
title: FA
description: API reference for FA.
---

# `FA`

**Module:** `opytimizer.optimizers.single_objective.swarm.fa`

A FA class, inherited from Optimizer.

This is the designed class to define FA-related
variables and methods.

References:
    X.-S. Yang. Firefly algorithms for multimodal optimization.
    International symposium on stochastic algorithms (2009).

## Constructor

```python
FA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, n_iterations: int) -> None
```

Wraps Firefly Algorithm over all agents and variables (eq. 3-9).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
