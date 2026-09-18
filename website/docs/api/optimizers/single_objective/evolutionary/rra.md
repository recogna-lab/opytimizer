---
title: RRA
description: API reference for RRA.
---

# `RRA`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.rra`

An RRA class, inherited from Optimizer.

This is the designed class to define RRA-related
variables and methods.

References:
    F. Merrikh-Bayat.
    The runner-root algorithm: A metaheuristic for solving unimodal and
    multimodal optimization problems inspired by runners and roots of plants in nature.
    Applied Soft Computing (2015).

## Constructor

```python
RRA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Runner-Root Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
