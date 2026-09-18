---
title: OBMOCE
description: API reference for OBMOCE.
---

# `OBMOCE`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.moce`

Opposition Learning-based Multi-Objective Chaotic Evolution (OBMOCE).

Extends MOCE by integrating the OBL mechanism (CEcOB variant):
for each individual, generates a chaotic vector and its opposite,
then selects the next generation from the combined pool of
current population + chaotic + opposite using non-dominated
sorting and crowding distance.

Pool size per generation: 3 x PS.

References:
   Li, T., & Pei, Y. (2025). Opposition-based chaotic evolution for optimization.
   Scientific Reports, 15(1), 22718.

## Constructor

```python
OBMOCE(params: Optional[Dict[str, Any]] = None, DR: float = 0.5, CR: float = 0.9, chaotic_system: Literal['logistic', 'gauss', 'tent', 'henon'] = 'logistic')
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
