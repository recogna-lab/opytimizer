---
title: CS
description: API reference for CS.
---

# `CS`

**Module:** `opytimizer.optimizers.single_objective.swarm.cs`

A CS class, inherited from Optimizer.

This is the designed class to define CS-related
variables and methods.

References:
    X.-S. Yang and D. Suash. Cuckoo search via Lévy flights.
    World Congress on Nature & Biologically Inspired Computing (2009).

## Constructor

```python
CS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Cuckoo Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
