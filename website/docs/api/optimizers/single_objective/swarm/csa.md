---
title: CSA
description: API reference for CSA.
---

# `CSA`

**Module:** `opytimizer.optimizers.single_objective.swarm.csa`

A CSA class, inherited from Optimizer.

This is the designed class to define CSA-related
variables and methods.

References:
    A. Askarzadeh. A novel metaheuristic method for
    solving constrained engineering optimization problems: Crow search algorithm.
    Computers & Structures (2016).

## Constructor

```python
CSA(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object containing meta-information. |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace) -> None
```

Wraps Crow Search Algorithm over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
