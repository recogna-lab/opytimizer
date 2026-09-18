---
title: FPA
description: API reference for FPA.
---

# `FPA`

**Module:** `opytimizer.optimizers.single_objective.swarm.fpa`

A FPA class, inherited from Optimizer.

This is the designed class to define FPA-related
variables and methods.

References:
    X.-S. Yang. Flower pollination algorithm for global optimization.
    International conference on unconventional computing and natural computation (2012).

## Constructor

```python
FPA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Flower Pollination Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
