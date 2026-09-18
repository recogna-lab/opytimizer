---
title: QuantumCDO
description: API reference for QuantumCDO.
---

# `QuantumCDO`

**Module:** `opytimizer.optimizers.single_objective.science.cdo`

Quantum Chernobyl Disaster Optimizer.

This variant implements quantum-inspired mechanisms for better exploration.

## Constructor

```python
QuantumCDO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `quantum_position`

```python
quantum_position(self, center: numpy.ndarray, space: opytimizer.core.space._SingleObjectiveSpace) -> numpy.ndarray
```

Generate quantum position around center.

Returns:
    Quantum position

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `center` | `numpy.ndarray` |  | Center position for quantum cloud |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space object containing bounds |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, function: opytimizer.core.function.Function, iteration: int, n_iterations: int) -> None
```

Updates using quantum-inspired mechanisms.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information |
| `function` | `opytimizer.core.function.Function` |  | Objective function |
| `iteration` | `int` |  | Current iteration |
| `n_iterations` | `int` |  | Maximum iterations |
