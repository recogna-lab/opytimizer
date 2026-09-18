---
title: OBCE
description: API reference for OBCE.
---

# `OBCE`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.ce`

Opposition-Based Chaotic Evolution — Single-Objective (CEcOB variant).

Integrates Opposition-Based Learning (OBL) into the conventional CE
algorithm. For each individual, generates a chaotic vector and its
opposite vector, then keeps the best among the three candidates
(target, chaotic, chaotic-OB).


References:
    T. Li and Y. Pei, "Opposition-based chaotic evolution for optimization,"
    Scientific Reports, 15, 22718 (2025).

## Constructor

```python
OBCE(params: Optional[Dict[str, Any]] = None, DR: float = 0.5, CR: float = 0.9, chaotic_system: Literal['logistic', 'gauss', 'tent', 'henon'] = 'logistic')
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `DR` | `float` | `0.5` | — |
| `CR` | `float` | `0.9` | — |
| `chaotic_system` | `Literal['logistic', 'gauss', 'tent', 'henon']` | `'logistic'` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._SingleObjectiveSpace)
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | — |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function)
```

Evaluates the search space according to the objective function.

If you need a specific evaluate method, please re-implement
it on child's class.

Also, note that function only accept arguments that are
found on Opytimizer class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

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
