---
title: SimulatedAnnealingSelection
description: API reference for SimulatedAnnealingSelection.
---

# `SimulatedAnnealingSelection`

**Module:** `opytimizer.hyperheuristics.selection_strategy`

Simulated Annealing selection strategy.

## Constructor

```python
SimulatedAnnealingSelection(initial_temperature: float = 100.0, cooling_rate: float = 0.95) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `initial_temperature` | `float` | `100.0` | — |
| `cooling_rate` | `float` | `0.95` | — |

## Methods

### `reset`

```python
reset(self) -> None
```

Reset temperature and iteration counter.

### `select`

```python
select(self, optimizers: List[Any], performance_history: Dict[str, List[float]]) -> Any
```

Select optimizer using Simulated Annealing.

Returns:
    (Any): Selected optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizers` | `List[Any]` |  | List of available optimizers. |
| `performance_history` | `Dict[str, List[float]]` |  | Performance history for each optimizer. |
