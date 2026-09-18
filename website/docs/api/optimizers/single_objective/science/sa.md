---
title: SA
description: API reference for SA.
---

# `SA`

**Module:** `opytimizer.optimizers.single_objective.science.sa`

A SA class, inherited from Optimizer.

This is the designed class to define SA-related
variables and methods.

References:
    A. Khachaturyan, S. Semenovsovskaya and B. Vainshtein.
    The thermodynamic approach to the structure analysis of crystals.
    Acta Crystallographica (1981).

## Constructor

```python
SA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Wraps Simulated Annealing over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `function` | `opytimizer.core.function.Function` |  | A function object. |
