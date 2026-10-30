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
| `params` | `Optional[Dict[str, Any]]` | `None` | Contains key-value parameters to the meta-heuristics. |
| `crossover_operator` |  | `None` | Tensorized crossover operator. Defaults to SBX with eta = 30 and crossover probability 1.0, as in Cheng et al. (2016). |
| `mutation_operator` |  | `None` | Tensorized mutation operator. Defaults to the polynomial mutation with eta = 20 and rate 1 / n_variables. |
| `reference_vectors` | `numpy.ndarray` | `None` | Reference vectors, one per agent. |
| `max_generations` | `int` | `250` | Number of generations t_max of the run, used by the penalty function and the adaptation frequency. It should match the stopping criterion. |
| `alpha` | `Union[float, int]` | `2.0` | Rate of change of the penalty function. |
| `fr` | `float` | `0.1` | Frequency of the reference vector adaptation (fraction of t_max). |

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
