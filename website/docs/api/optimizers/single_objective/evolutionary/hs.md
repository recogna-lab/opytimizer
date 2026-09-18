---
title: HS
description: API reference for HS.
---

# `HS`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.hs`

A HS class, inherited from Optimizer.

This is the designed class to define HS-related
variables and methods.

References:
    Z. W. Geem, J. H. Kim, and G. V. Loganathan.
    A new heuristic optimization algorithm: Harmony search. Simulation (2001).

## Constructor

```python
HS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Harmony Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
