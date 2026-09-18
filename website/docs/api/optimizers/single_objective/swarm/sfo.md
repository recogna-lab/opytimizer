---
title: SFO
description: API reference for SFO.
---

# `SFO`

**Module:** `opytimizer.optimizers.single_objective.swarm.sfo`

A SFO class, inherited from Optimizer.

This is the designed class to define SFO-related
variables and methods.

References:
    S. Shadravan, H. Naji and V. Bardsiri.
    The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm
    for solving constrained engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2019).

## Constructor

```python
SFO(params: Optional[Dict[str, Any]] = None) -> None
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
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int) -> None
```

Wraps Sailfish Optimizer over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `iteration` | `int` |  | Current iteration. |
