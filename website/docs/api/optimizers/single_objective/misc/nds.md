---
title: NDS
description: API reference for NDS.
---

# `NDS`

**Module:** `opytimizer.optimizers.single_objective.misc.nds`

An NDS class, inherited from Optimizer.

This is the designed class to define NDS-related
variables and methods.

References:
    P. Godfrey, R. Shipley and J. Gryz.
    Algorithms and Analyses for Maximal Vector Computation.
    The VLDB Journal (2007).

## Constructor

```python
NDS(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Non-Dominated Sorting over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
