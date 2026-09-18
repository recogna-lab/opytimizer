---
title: CMAESCuda
description: API reference for CMAESCuda.
---

# `CMAESCuda`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.cmaes`

A CMAESCuda class, inherited from CMAESTensor.

This class offers a highly accelerated GPU-friendly implementation
using CuPy RawKernels for CMA-ES Population Generation.

## Constructor

```python
CMAESCuda(params: Optional[Dict[str, Any]] = None, **kwargs) -> None
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

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveTensorSpace) -> None
```

Wraps CMA-ES over all agents and variables using CUDA Raw Kernels.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveTensorSpace` |  | Space containing agents and update-related information. |
