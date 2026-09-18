---
title: CI
description: API reference for CI.
---

# `CI`

**Module:** `opytimizer.optimizers.single_objective.social.ci`

A CI class, inherited from Optimizer.

This is the designed class to define CI-related
variables and methods.

References:
    A. J. Kulkarni, I. P. Durugkar, M. Kumar. Cohort Intelligence: A Self Supervised Learning Behavior.
    IEEE International Conference on Systems, Man, and Cybernetics (2013).

## Constructor

```python
CI(params: Optional[Dict[str, Any]] = None) -> None
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

Wraps Cohort Intelligence over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
