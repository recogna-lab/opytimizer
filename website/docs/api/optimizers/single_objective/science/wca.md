---
title: WCA
description: API reference for WCA.
---

# `WCA`

**Module:** `opytimizer.optimizers.single_objective.science.wca`

A WCA class, inherited from Optimizer.

This is the designed class to define WCA-related
variables and methods.

References:
    H. Eskandar.
    Water cycle algorithm – A novel metaheuristic optimization method for
    solving constrained engineering optimization problems.
    Computers & Structures (2012).

## Constructor

```python
WCA(params: Optional[Dict[str, Any]] = None) -> None
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
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, n_iterations: int) -> None
```

Wraps Water Cycle Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
