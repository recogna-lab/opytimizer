---
title: OSA
description: API reference for OSA.
---

# `OSA`

**Module:** `opytimizer.optimizers.single_objective.population.osa`

An OSA class, inherited from Optimizer.

This is the designed class to define OSA-related
variables and methods.

References:
    M. Jain, S. Maurya, A. Rani and V. Singh.
    Owl search algorithm: A novelnature-inspired heuristic paradigm for global optimization.
    Journal of Intelligent & Fuzzy Systems (2018).

## Constructor

```python
OSA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int, n_iterations: int) -> None
```

Wraps Owl Search Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
| `n_iterations` | `int` |  | Maximum number of iterations. |
