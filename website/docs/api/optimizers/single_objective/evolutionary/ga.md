---
title: GA
description: API reference for GA.
---

# `GA`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.ga`

An GA class, inherited from Optimizer.

This is the designed class to define GA-related
variables and methods.

References:
    M. Mitchell. An introduction to genetic algorithms. MIT Press (1998).

## Constructor

```python
GA(params: Optional[Dict[str, Any]] = None, crossover_operator=None, mutation_operator=None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Wraps Genetic Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
