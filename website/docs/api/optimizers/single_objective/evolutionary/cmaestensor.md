---
title: CMAESTensor
description: API reference for CMAESTensor.
---

# `CMAESTensor`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.cmaes`

A CMAESTensor class, inherited from Optimizer and TensorizedOptimizer.

This is the designed class to define GPU/CPU-agnostic tensorized
CMA-ES-related variables and methods.

## Constructor

```python
CMAESTensor(params: Optional[Dict[str, Any]] = None, **kwargs) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `kwargs` |  |  | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._SingleObjectiveTensorSpace) -> None
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveTensorSpace` |  | A TensorSpace object containing meta-information. |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._SingleObjectiveTensorSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveTensorSpace` |  | A TensorSpace object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveTensorSpace) -> None
```

Wraps CMA-ES over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveTensorSpace` |  | Space containing agents and update-related information. |
