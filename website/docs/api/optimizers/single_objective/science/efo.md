---
title: EFO
description: API reference for EFO.
---

# `EFO`

**Module:** `opytimizer.optimizers.single_objective.science.efo`

An EFO class, inherited from Optimizer.

This is the designed class to define EFO-related
variables and methods.

References:
    H. Abedinpourshotorban et al.
    Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
    Swarm and Evolutionary Computation (2016).

## Constructor

```python
EFO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Wraps Electromagnetic Field Optimization over all agents and variables (eq. 1-4).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
