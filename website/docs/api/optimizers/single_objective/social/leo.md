---
title: LEO
description: API reference for LEO.
---

# `LEO`

**Module:** `opytimizer.optimizers.single_objective.social.leo`

A LEO class, inherited from Optimizer.

This is the designed class to define LEO-related
variables and methods.

References:
    P. Trojovsk`y, M. Dehghani, E. Trojovsk´a, and E. Milkova, “Language
    education optimization: A new human-based metaheuristic algorithm for
    solving optimization problems,” Computer Modeling in Engineering &
    Sciences, volume 136, issue: 2, 2023

## Constructor

```python
LEO(params: Optional[Dict[str, Any]] = None) -> None
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

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing a list of Agent objects, each with 'position' (NumPy array of shape (n, 1)) and 'fit' (scalar). |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
