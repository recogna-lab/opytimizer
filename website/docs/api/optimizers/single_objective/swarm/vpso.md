---
title: VPSO
description: API reference for VPSO.
---

# `VPSO`

**Module:** `opytimizer.optimizers.single_objective.swarm.pso`

A VPSO class, inherited from Optimizer.

This is the designed class to define VPSO-related
variables and methods.

References:
    W.-P. Yang. Vertical particle swarm optimization algorithm and its application in soft-sensor modeling.
    International Conference on Machine Learning and Cybernetics (2007).

## Constructor

```python
VPSO(params: Optional[Dict[str, Any]] = None) -> None
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
update(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Wraps Vertical Particle Swarm Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information. |
