---
title: BOA
description: API reference for BOA.
---

# `BOA`

**Module:** `opytimizer.optimizers.single_objective.swarm.boa`

A BOA class, inherited from Optimizer.

This is the designed class to define BOA-related
variables and methods.

References:
    S. Arora and S. Singh. Butterfly optimization algorithm: a novel approach for global optimization.
    Soft Computing (2019).

## Constructor

```python
BOA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Butterfly Optimization Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
