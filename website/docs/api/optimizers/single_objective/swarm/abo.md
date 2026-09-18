---
title: ABO
description: API reference for ABO.
---

# `ABO`

**Module:** `opytimizer.optimizers.single_objective.swarm.abo`

An ABO class, inherited from Optimizer.

This is the designed class to define ABO-related
variables and methods.

References:
    X. Qi, Y. Zhu and H. Zhang. A new meta-heuristic butterfly-inspired algorithm.
    Journal of Computational Science (2017).

## Constructor

```python
ABO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Artificial Butterfly Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
