---
title: ChaoticCDO
description: API reference for ChaoticCDO.
---

# `ChaoticCDO`

**Module:** `opytimizer.optimizers.single_objective.science.cdo`

Chaotic Chernobyl Disaster Optimizer.

This variant uses chaotic maps to improve the search behavior.

## Constructor

```python
ChaoticCDO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `get_chaotic_value`

```python
get_chaotic_value(self) -> float
```

Get chaotic value based on current map type.

### `logistic_map`

```python
logistic_map(self) -> float
```

Implements logistic map for chaos generation.

### `sine_map`

```python
sine_map(self) -> float
```

Implements sine map for chaos generation.

### `tent_map`

```python
tent_map(self) -> float
```

Implements tent map for chaos generation.

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Updates using chaotic values.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information |
| `function` | `opytimizer.core.function.Function` |  | Objective function |
| `iteration` | `int` |  | Current iteration |
| `n_iterations` | `int` |  | Maximum iterations |

### `update_map_selection`

```python
update_map_selection(self, success: bool) -> None
```

Update map selection based on success.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `success` | `bool` |  | — |
