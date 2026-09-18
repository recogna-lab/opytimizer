---
title: DOA
description: API reference for DOA.
---

# `DOA`

**Module:** `opytimizer.optimizers.single_objective.misc.doa`

A DOA class, inherited from Optimizer.

This is the designed class to define DOA-related
variables and methods.

References:
    F. Demir et al. A survival classification method for hepatocellular carcinoma patients
    with chaotic Darcy optimization method based feature selection.
    Medical Hypotheses (2020).

## Constructor

```python
DOA(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Darcy Optimization Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
