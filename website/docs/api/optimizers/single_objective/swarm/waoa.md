---
title: WAOA
description: API reference for WAOA.
---

# `WAOA`

**Module:** `opytimizer.optimizers.single_objective.swarm.waoa`

A WAOA class, inherited from Optimizer.

This is the designed class to define WAOA-related
variables and methods.

References:
    P. Trojovský and M. Dehghani. A new bio-inspired metaheuristic algorithm for
    solving optimization problems based on walruses behavior. Scientific Reports (2023).

## Constructor

```python
WAOA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object that will be evaluated. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int) -> None
```

Wraps Walrus Optimization Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
