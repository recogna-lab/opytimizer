---
title: AIG
description: API reference for AIG.
---

# `AIG`

**Module:** `opytimizer.optimizers.single_objective.science.aig`

An AIG class, inherited from Optimizer.

This is the designed class to define AIG-related
variables and methods.

References:
    P. Pijarski and P. Kacejko.
    A new metaheuristic optimization method: the algorithm of the innovative gunner (AIG).
    Engineering Optimization (2019).

## Constructor

```python
AIG(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Algorithm of the Innovative Gunner over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
