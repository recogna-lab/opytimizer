---
title: MOA
description: API reference for MOA.
---

# `MOA`

**Module:** `opytimizer.optimizers.single_objective.science.moa`

An MOA class, inherited from Optimizer.

This is the designed class to define MOA-related
variables and methods.

References:
    M.-H. Tayarani and M.-R. Akbarzadeh. Magnetic-inspired optimization algorithms: Operators and structures.
    Swarm and Evolutionary Computation (2014).

## Constructor

```python
MOA(params: Optional[Dict[str, Any]] = None) -> None
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
update(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Wraps Magnetic Optimization Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
