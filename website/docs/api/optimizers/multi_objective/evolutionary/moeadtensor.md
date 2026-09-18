---
title: MOEADTensor
description: API reference for MOEADTensor.
---

# `MOEADTensor`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.moead`

Tensorized implementation of MOEA/D
based on the following work:
Z. Liang, H. Li, N. Yu, K. Sun and R. Cheng, "Bridging Evolutionary Multiobjective Optimization and GPU Acceleration via Tensorization,"
in IEEE Transactions on Evolutionary Computation, vol. 30, no. 1, pp. 420-434, Feb. 2026, doi: 10.1109/TEVC.2025.3555605.

## Constructor

```python
MOEADTensor(params: Optional[Dict[str, Any]] = None, crossover_operator=None, mutation_operator=None, weight_vectors=None, decomposition_method: opytimizer.math.aggregation._BaseAggregation = None, neighborhood_size: int = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |
| `weight_vectors` |  | `None` | — |
| `decomposition_method` | `opytimizer.math.aggregation._BaseAggregation` | `None` | — |
| `neighborhood_size` | `int` | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._MultiObjectiveTensorSpace, **kwargs) -> None
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveTensorSpace` |  | — |
| `kwargs` |  |  | — |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._MultiObjectiveTensorSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveTensorSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveTensorSpace, function: opytimizer.core.function.Function) -> None
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveTensorSpace` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |
