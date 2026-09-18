---
title: MultiReactorCDO
description: API reference for MultiReactorCDO.
---

# `MultiReactorCDO`

**Module:** `opytimizer.optimizers.single_objective.science.cdo`

Multi-Reactor Chernobyl Disaster Optimizer.

This variant uses multiple sub-populations (reactors) that occasionally exchange information.

## Constructor

```python
MultiReactorCDO(params: Optional[Dict[str, Any]] = None) -> None
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

Compiles additional information for each reactor.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing meta-information. |

### `exchange_information`

```python
exchange_information(self) -> None
```

Exchange best solutions between reactors.

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Updates using multiple reactors.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information |
| `function` | `opytimizer.core.function.Function` |  | Objective function |
| `iteration` | `int` |  | Current iteration |
| `n_iterations` | `int` |  | Maximum iterations |
