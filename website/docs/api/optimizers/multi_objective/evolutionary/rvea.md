---
title: RVEA
description: API reference for RVEA.
---

# `RVEA`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.rvea`

RVEA class, inherited from MultiObjectiveOptimizer.

References:
    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization,"
    in IEEE Transactions on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, Oct. 2016, doi: 10.1109/TEVC.2016.2519378.

## Constructor

```python
RVEA(params: Optional[Dict[str, Any]] = None, crossover_operator=None, mutation_operator=None, reference_vectors: numpy.ndarray = None, max_generations: int = 250, alpha: Union[float, int] = 2.0, fr: float = 0.1, **kwargs)
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
| `kwargs` |  |  | — |

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
evaluate(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function)
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function)
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |
