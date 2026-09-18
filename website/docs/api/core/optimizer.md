---
title: Optimizer
description: API reference for Optimizer.
---

# `Optimizer`

**Module:** `opytimizer.core.optimizer`

An Optimizer class that holds meta-heuristics-related properties
and methods.

## Constructor

```python
Optimizer() -> None
```

## Methods

### `build`

```python
build(self, params: Dict[str, Any]) -> None
```

Builds the object by creating its parameters.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Dict[str, Any]` |  | Key-value parameters to the meta-heuristic. |

### `compile`

```python
compile(self, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._SingleObjectiveTensorSpace, opytimizer.core.space._MultiObjectiveSpace, opytimizer.core.space._MultiObjectiveTensorSpace]) -> None
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._SingleObjectiveTensorSpace, opytimizer.core.space._MultiObjectiveSpace, opytimizer.core.space._MultiObjectiveTensorSpace]` |  | — |

### `evaluate`

```python
evaluate(self, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._SingleObjectiveTensorSpace], function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

If you need a specific evaluate method, please re-implement
it on child's class.

Also, note that function only accept arguments that are
found on Opytimizer class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._SingleObjectiveTensorSpace]` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._SingleObjectiveTensorSpace, opytimizer.core.space._MultiObjectiveSpace, opytimizer.core.space._MultiObjectiveTensorSpace], function: opytimizer.core.function.Function) -> None
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._SingleObjectiveTensorSpace, opytimizer.core.space._MultiObjectiveSpace, opytimizer.core.space._MultiObjectiveTensorSpace]` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |
