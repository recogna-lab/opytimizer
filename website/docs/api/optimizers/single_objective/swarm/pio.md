---
title: PIO
description: API reference for PIO.
---

# `PIO`

**Module:** `opytimizer.optimizers.single_objective.swarm.pio`

A PIO class, inherited from Optimizer.

This is the designed class to define PIO-related
variables and methods.

References:
    H. Duan and P. Qiao.
    Pigeon-inspired optimization:a new swarm intelligence optimizerfor air robot path planning.
    International Journal of IntelligentComputing and Cybernetics (2014).

## Constructor

```python
PIO(params: Optional[Dict[str, Any]] = None) -> None
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

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveSpace, iteration: int) -> None
```

Wraps Pigeon-Inspired Optimization over all agents and variables.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveSpace` |  | Space containing agents and update-related information. |
| `iteration` | `int` |  | Current iteration. |
