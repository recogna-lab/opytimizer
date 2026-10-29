---
title: NSGA2
description: API reference for NSGA2.
---

# `NSGA2`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.nsga2`

NSGA2 class, inherited from MultiObjectiveOptimizer.

References:
    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
    multiobjective genetic algorithm: NSGA-II," IEEE Trans. Evol. Comput.,
    vol. 6, no. 2, pp. 182-197, 2002.

## Constructor

```python
NSGA2(params: 'Dict' = None, crossover_operator=None, mutation_operator=None) -> 'None'
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Dict` | `None` | Contains key-value parameters to the meta-heuristics. |
| `crossover_operator` |  | `None` | Crossover operator applied to pairs of parents. |
| `mutation_operator` |  | `None` | Mutation operator applied to every child. |

## Methods

### `compile`

```python
compile(self, space: '_MultiObjectiveSpace') -> 'None'
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveSpace` |  | A MultiObjectiveSpace object. |

### `evaluate`

```python
evaluate(self, space, function)
```

Evaluates the initial population and computes its ranks and crowding.

Offspring are evaluated inside `update`, so this method replaces itself
with a no-op after its first call.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` |  |  | A MultiObjectiveSpace object. |
| `function` |  |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: '_MultiObjectiveSpace', function) -> 'None'
```

Performs one generation: creates, evaluates and selects the offspring.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveSpace` |  | A MultiObjectiveSpace object. |
| `function` |  |  | A Function object that will be used as the objective function. |
