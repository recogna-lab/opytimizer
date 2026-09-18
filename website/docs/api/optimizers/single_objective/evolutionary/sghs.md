---
title: SGHS
description: API reference for SGHS.
---

# `SGHS`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.hs`

A SGHS class, inherited from HS.

This is the designed class to define SGHS-related
variables and methods.

References:
    Q.-K. Pan, P. Suganthan, M. Tasgetiren and J. Liang.
    A self-adaptive global best harmony search algorithm for continuous optimization problems.
    Applied Mathematics and Computation (2010).

## Constructor

```python
SGHS(params: Optional[Dict[str, Any]] = None) -> None
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
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Wraps Self-Adaptive Global-Best Harmony Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
