---
title: NSGA2Tensor
description: API reference for NSGA2Tensor.
---

# `NSGA2Tensor`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.nsga2`

Tensorized NSGA-II, following the general tensorization methodology of:

Z. Liang, H. Li, N. Yu, K. Sun, and R. Cheng, "Bridging Evolutionary
Multiobjective Optimization and GPU Acceleration via Tensorization,"
IEEE Trans. Evol. Comput., vol. 30, no. 1, pp. 420-434, Feb. 2026.

## Constructor

```python
NSGA2Tensor(params: 'dict' = None, crossover_operator=None, mutation_operator=None) -> 'None'
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `dict` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |

## Methods

### `compile`

```python
compile(self, space: '_MultiObjectiveTensorSpace') -> 'None'
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveTensorSpace` |  | — |

### `evaluate`

```python
evaluate(self, space: '_MultiObjectiveTensorSpace', function) -> 'None'
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveTensorSpace` |  | A Space object that will be evaluated. |
| `function` |  |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: '_MultiObjectiveTensorSpace', function) -> 'None'
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveTensorSpace` |  | — |
| `function` |  |  | — |
