---
title: CMAES
description: API reference for CMAES.
---

# `CMAES`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.cmaes`

A CMAES class, inherited from Optimizer.

This is the designed class to define CMA-ES-related
variables and methods.

References:
    N. Hansen. The CMA evolution strategy: a comparing review.
    Towards a new evolutionary computation (2006).

## Constructor

```python
CMAES(params: Optional[Dict[str, Any]] = None, **kwargs) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `kwargs` |  |  | — |

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

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Wraps CMA-ES over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
