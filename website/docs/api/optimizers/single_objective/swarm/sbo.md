---
title: SBO
description: API reference for SBO.
---

# `SBO`

**Module:** `opytimizer.optimizers.single_objective.swarm.sbo`

A SBO class, inherited from Optimizer.

This is the designed class to define SBO-related
variables and methods.

References:
    S. H. S. Moosavi and V. K. Bardsiri.
    Satin bowerbird optimizer: a new optimization algorithm to optimize ANFIS
    for software development effort estimation.
    Engineering Applications of Artificial Intelligence (2017).

## Constructor

```python
SBO(params: Optional[Dict[str, Any]] = None) -> None
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
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Wraps Satin Bowerbird Optimizer over all agents and variables (eq. 1-7).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
