---
title: PSOTensor
description: API reference for PSOTensor.
---

# `PSOTensor`

**Module:** `opytimizer.optimizers.single_objective.swarm.pso`

GPU/CPU-agnostic, fully tensorized implementation of PSO (single-objective),

## Constructor

```python
PSOTensor(params: Optional[Dict[str, Any]] = None, **kwargs) -> None
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
