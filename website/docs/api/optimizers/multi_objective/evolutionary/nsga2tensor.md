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
| `params` | `dict` | `None` | Contains key-value parameters to the meta-heuristics. |
| `crossover_operator` |  | `None` | Tensorized crossover operator applied to the parents. |
| `mutation_operator` |  | `None` | Tensorized mutation operator applied to the children. |

## Methods

### `compile`

```python
compile(self, space: '_MultiObjectiveTensorSpace') -> 'None'
```

Allocates the population buffers and binds the optimizer to the backend.

`P_tensor` / `F_tensor` hold `2 * n_agents` rows: the current population in
the first half and the offspring of the generation in the second one.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveTensorSpace` |  | A tensorized MultiObjectiveSpace object. |

### `evaluate`

```python
evaluate(self, space: '_MultiObjectiveTensorSpace', function) -> 'None'
```

Evaluates the initial population and computes its ranks and crowding.

Offspring are evaluated inside `update`, so this method replaces itself
with a no-op after its first call.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveTensorSpace` |  | A tensorized MultiObjectiveSpace object. |
| `function` |  |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: '_MultiObjectiveTensorSpace', function) -> 'None'
```

Performs one generation: creates, evaluates and selects the offspring.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveTensorSpace` |  | A tensorized MultiObjectiveSpace object. |
| `function` |  |  | A Function object that will be used as the objective function. |
