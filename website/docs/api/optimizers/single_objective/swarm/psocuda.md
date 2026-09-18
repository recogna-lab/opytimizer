---
title: PSOCuda
description: API reference for PSOCuda.
---

# `PSOCuda`

**Module:** `opytimizer.optimizers.single_objective.swarm.pso`

GPU-friendly, fully tensorized implementation of PSO (single-objective).

All particle state -- current position, velocity, personal best
(local) position/fitness, and the running global best -- lives on the
GPU as `xp` tensors for the entire run. There is NO host `<->` device
synchronization inside `evaluate`/`update`; `space.agents` is read
exactly once, at `compile` time, to seed the initial positions. The
only place data is pulled back to the host is `sync`, meant
to be called a single time, after the optimization loop has finished.

## Constructor

```python
PSOCuda(params: Optional[Dict[str, Any]] = None, **kwargs) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |
| `kwargs` |  |  | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._SingleObjectiveTensorSpace) -> None
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveTensorSpace` |  | — |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._SingleObjectiveTensorSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

If you need a specific evaluate method, please re-implement
it on child's class.

Also, note that function only accept arguments that are
found on Opytimizer class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveTensorSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._SingleObjectiveTensorSpace) -> None
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._SingleObjectiveTensorSpace` |  | — |
