---
title: PVS
description: API reference for PVS.
---

# `PVS`

**Module:** `opytimizer.optimizers.single_objective.population.pvs`

A PVS class, inherited from Optimizer.

This is the designed class to define PVS-related
variables and methods.

References:
    P. Savsani and V. Savsani. Passing vehicle search (PVS): A novel metaheuristic algorithm.
    Applied Mathematical Modelling (2016).

## Constructor

```python
PVS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Passing Vehicle Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
