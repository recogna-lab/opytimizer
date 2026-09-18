---
title: MOEAD
description: API reference for MOEAD.
---

# `MOEAD`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.moead`

MOEAD class, inherited from Optimizer.

References:
    Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
    IEEE Transactions on evolutionary computation, 11(6), 712-731.

## Constructor

```python
MOEAD(params: Optional[Dict[str, Any]] = None, crossover_operator=None, mutation_operator=None, weight_vectors=None, neighborhood_size: int = None, decomposition_method: opytimizer.math.aggregation._BaseAggregation = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |
| `weight_vectors` |  | `None` | — |
| `neighborhood_size` | `int` | `None` | — |
| `decomposition_method` | `opytimizer.math.aggregation._BaseAggregation` | `None` | — |

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

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |
