---
title: RPSO
description: API reference for RPSO.
---

# `RPSO`

**Module:** `opytimizer.optimizers.single_objective.swarm.pso`

An RPSO class, inherited from Optimizer.

This is the designed class to define RPSO-related
variables and methods.

References:
    M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
    Harnessing Particle Swarm Optimization Through Relativistic Velocity.
    IEEE Congress on Evolutionary Computation (2020).

## Constructor

```python
RPSO(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Relativistic Particle Swarm Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Union[_SingleObjectiveSpace, _MultiObjectiveSpace] containing agents and update-related information. |
