---
title: CE
description: API reference for CE.
---

# `CE`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.ce`

A CE class, inherited from Optimizer.

This is the designed class to define CE-related
variables and methods.

References:
    Pei, Y. (2020, October). Chaotic evolution algorithm with elite strategy in single-objective and multi-objective optimization.
    In 2020 IEEE international conference on systems, man, and cybernetics (SMC) (pp. 579-584). IEEE.

## Constructor

```python
CE(params: Optional[Dict[str, Any]] = None, DR: float = 0.7, CR: float = 0.7, p: Tuple[float, float] = (0.02, 0.1), jump: int = 10, chaotic_system: Literal['logistic', 'gauss', 'tent', 'henon'] = 'tent')
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `DR` | `float` | `0.7` | — |
| `CR` | `float` | `0.7` | — |
| `p` | `Tuple[float, float]` | `(0.02, 0.1)` | — |
| `jump` | `int` | `10` | — |
| `chaotic_system` | `Literal['logistic', 'gauss', 'tent', 'henon']` | `'tent'` | — |

## Methods

### `compile`

```python
compile(self, space)
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` |  |  | — |

### `evaluate`

```python
evaluate(self, space, function)
```

Evaluates the search space according to the objective function.

If you need a specific evaluate method, please re-implement
it on child's class.

Also, note that function only accept arguments that are
found on Opytimizer class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` |  |  | A Space object that will be evaluated. |
| `function` |  |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function)
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |
