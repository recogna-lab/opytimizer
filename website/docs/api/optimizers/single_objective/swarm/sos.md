---
title: SOS
description: API reference for SOS.
---

# `SOS`

**Module:** `opytimizer.optimizers.single_objective.swarm.sos`

An SOS class, inherited from Optimizer.

This is the designed class to define SOS-related
variables and methods.

References:
    M.-Y. Cheng and D. Prayogo. Symbiotic Organisms Search: A new metaheuristic optimization algorithm.
    Computers & Structures (2014).

## Constructor

```python
SOS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Symbiotic Organisms Search over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
