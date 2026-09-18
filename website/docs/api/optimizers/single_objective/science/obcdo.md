---
title: OBCDO
description: API reference for OBCDO.
---

# `OBCDO`

**Module:** `opytimizer.optimizers.single_objective.science.cdo`

Opposition-Based Chernobyl Disaster Optimizer.

This variant implements multiple Opposition-Based Learning strategies:
- Basic OBL (BOBL)
- Quasi OBL (QOBL)
- Generalized OBL (GOBL)
- Partial OBL (POBL)
- Center-Based OBL (COBL)
- Enhanced OBL (EOBL)
- Time-Varying OBL (TVOBL)
- Elite OBL (Elite-OBL)

## Constructor

```python
OBCDO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `adapt_obl_rate`

```python
adapt_obl_rate(self, success_rate: float) -> None
```

Adapt opposition rate based on success rate.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `success_rate` | `float` |  | Rate of successful oppositions |

### `get_center_opposite_position`

```python
get_center_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Center-Based Opposition-Based Learning (COBL).

Returns:
    Center-based opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `get_elite_opposite_position`

```python
get_elite_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Elite Opposition-Based Learning (Elite-OBL).

Returns:
    Elite-based opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `get_enhanced_opposite_position`

```python
get_enhanced_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Enhanced Opposition-Based Learning (EOBL).

Returns:
    Enhanced opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `get_generalized_opposite_position`

```python
get_generalized_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Generalized Opposition-Based Learning (GOBL).

Returns:
    Generalized opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `get_opposite_position`

```python
get_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Basic Opposition-Based Learning (BOBL).

Returns:
    Opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `get_partial_opposite_position`

```python
get_partial_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Partial Opposition-Based Learning (POBL).

Returns:
    Partial opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `get_quasi_opposite_position`

```python
get_quasi_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Quasi Opposition-Based Learning (QOBL).

Returns:
    Quasi-opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `get_time_varying_opposite_position`

```python
get_time_varying_opposite_position(self, position: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int, n_iterations: int) -> numpy.ndarray
```

Time-Varying Opposition-Based Learning (TVOBL).

Returns:
    Time-varying opposite position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `position` | `numpy.ndarray` |  | Current position |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |
| `iteration` | `int` |  | Current iteration |
| `n_iterations` | `int` |  | Maximum iterations |

### `select_strategy`

```python
select_strategy(self, iteration: int, n_iterations: int) -> str
```

Select OBL strategy based on current state.

Returns:
    Selected strategy name

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `iteration` | `int` |  | Current iteration |
| `n_iterations` | `int` |  | Maximum iterations |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Updates using Opposition-Based Learning.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information |
| `function` | `opytimizer.core.function.Function` |  | Objective function |
| `iteration` | `int` |  | Current iteration |
| `n_iterations` | `int` |  | Maximum iterations |

### `update_elite_solutions`

```python
update_elite_solutions(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function) -> None
```

Update elite solutions pool.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |
