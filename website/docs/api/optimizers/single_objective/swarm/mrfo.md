---
title: MRFO
description: API reference for MRFO.
---

# `MRFO`

**Module:** `opytimizer.optimizers.single_objective.swarm.mrfo`

An MRFO class, inherited from Optimizer.

This is the designed class to define MRFO-related
variables and methods.

References:
    W. Zhao, Z. Zhang and L. Wang.
    Manta Ray Foraging Optimization: An effective bio-inspired optimizer for engineering applications.
    Engineering Applications of Artificial Intelligence (2020).

## Constructor

```python
MRFO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Wraps Manta Ray Foraging Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
