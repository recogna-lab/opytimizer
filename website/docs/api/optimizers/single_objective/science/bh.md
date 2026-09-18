---
title: BH
description: API reference for BH.
---

# `BH`

**Module:** `opytimizer.optimizers.single_objective.science.bh`

A BH class, inherited from Optimizer.

This is the designed class to define BH-related
variables and methods.

References:
    A. Hatamlou. Black hole: A new heuristic optimization approach for data clustering.
    Information Sciences (2013).

## Constructor

```python
BH(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Black Hole over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
