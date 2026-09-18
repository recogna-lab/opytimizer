---
title: MultiObjectiveOptimizer
description: API reference for MultiObjectiveOptimizer.
---

# `MultiObjectiveOptimizer`

**Module:** `opytimizer.core.optimizer`

A MultiObjectiveOptimizer class that holds multi-objective meta-heuristics-related
properties and methods.

## Constructor

```python
MultiObjectiveOptimizer() -> None
```

## Methods

### `evaluate`

```python
evaluate(self, space: Union[opytimizer.core.space._MultiObjectiveSpace, opytimizer.core.space._MultiObjectiveTensorSpace], function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._MultiObjectiveSpace, opytimizer.core.space._MultiObjectiveTensorSpace]` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |
