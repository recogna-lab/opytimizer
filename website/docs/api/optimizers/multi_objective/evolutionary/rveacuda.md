---
title: RVEACuda
description: API reference for RVEACuda.
---

# `RVEACuda`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.rvea`

GPU-friendly, fully tensorized implementation of RVEA.

Identical to `RVEATensor` except for the grouped-argmin step, which can run
as a single hand-written CUDA kernel.

## Constructor

```python
RVEACuda(params: Optional[Dict[str, Any]] = None, crossover_operator=None, mutation_operator=None, reference_vectors: numpy.ndarray = None, max_generations: int = 250, alpha: Union[float, int] = 2.0, fr: float = 0.1)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |
| `reference_vectors` | `numpy.ndarray` | `None` | — |
| `max_generations` | `int` | `250` | — |
| `alpha` | `Union[float, int]` | `2.0` | — |
| `fr` | `float` | `0.1` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._MultiObjectiveSpace)
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | — |
