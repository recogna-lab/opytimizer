---
title: MOEAD_DE
description: API reference for MOEAD_DE.
---

# `MOEAD_DE`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.moead`

MOEA/D-DE class, inherited from Optimizer.

References:
    Li, H., & Zhang, Q. (2008). Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II.
    IEEE transactions on evolutionary computation, 13(2), 284-302.

## Constructor

```python
MOEAD_DE(params: Optional[Dict[str, Any]] = None, CR: Union[float, int] = 1.0, nr: int = 2, F: Union[float, int] = 0.5, neighborhood_prob: float = 0.9, mutation_operator=None, weight_vectors=None, decomposition_function: opytimizer.math.aggregation._BaseAggregation = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `CR` | `Union[float, int]` | `1.0` | — |
| `nr` | `int` | `2` | — |
| `F` | `Union[float, int]` | `0.5` | — |
| `neighborhood_prob` | `float` | `0.9` | — |
| `mutation_operator` |  | `None` | — |
| `weight_vectors` |  | `None` | — |
| `decomposition_function` | `opytimizer.math.aggregation._BaseAggregation` | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._MultiObjectiveSpace, **kwargs) -> None
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | A Space object containing meta-information. |
| `kwargs` |  |  | — |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the fitness of the agents.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | Space containing agents and evaluation-related information. |
| `function` | `opytimizer.core.function.Function` |  | Function to evaluate the fitness of the agents. |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Updates the population using MOEA/D.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | Function to evaluate the fitness of the agents. |
