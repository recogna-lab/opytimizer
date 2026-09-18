---
title: GP
description: API reference for GP.
---

# `GP`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.gp`

A GP class, inherited from Optimizer.

Adapted to support Strongly-Typed Genetic Programming using Tree/Graph models.

## Constructor

```python
GP(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `evaluate`

```python
evaluate(self, space: opytimizer.core.graph.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.graph.space._SingleObjectiveSpace` |  | A TreeSpace object. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: opytimizer.core.graph.space._SingleObjectiveSpace) -> None
```

Wraps Genetic Programming over all trees and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.graph.space._SingleObjectiveSpace` |  | TreeSpace containing agents and update-related information. |
