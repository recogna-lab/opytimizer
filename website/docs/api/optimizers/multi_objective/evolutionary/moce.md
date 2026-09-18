---
title: MOCE
description: API reference for MOCE.
---

# `MOCE`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.moce`

Multi-Objective Chaotic Evolution (MOCE).

Uses chaotic ergodicity combined with non-dominated sorting
and crowding distance selection (NSGA-II style) for
multi-objective optimization.

References:
    Y. Pei, "Chaotic Evolution Algorithm with Elite Strategy
    in Single-objective and Multi-objective Optimization,"
    2020 IEEE International Conference on Systems, Man,
    and Cybernetics (SMC), Toronto, Canada, 2020,
    pp. 579-584.

## Constructor

```python
MOCE(params: Optional[Dict[str, Any]] = None, DR: float = 0.7, CR: float = 0.7, chaotic_system: Literal['logistic', 'gauss', 'tent', 'henon'] = 'logistic')
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `DR` | `float` | `0.7` | — |
| `CR` | `float` | `0.7` | — |
| `chaotic_system` | `Literal['logistic', 'gauss', 'tent', 'henon']` | `'logistic'` | — |

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
