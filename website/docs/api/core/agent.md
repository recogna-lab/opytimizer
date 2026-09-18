---
title: Agent
description: API reference for Agent.
---

# `Agent`

**Module:** `opytimizer.core.agent`

An Agent class for all optimization techniques.

## Constructor

```python
Agent(n_variables: int, n_dimensions: int, n_objectives: int, lower_bound: List[Union[int, float]], upper_bound: List[Union[int, float]], mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` |  | — |
| `n_dimensions` | `int` |  | — |
| `n_objectives` | `int` |  | — |
| `lower_bound` | `List[Union[int, float]]` |  | — |
| `upper_bound` | `List[Union[int, float]]` |  | — |
| `mapping` | `Optional[List[str]]` | `None` | — |
| `env` | `opytimizer.core.environment.Environment` | `None` | — |

## Methods

### `clip_by_bound`

```python
clip_by_bound(self) -> None
```

Clips the agent's decision variables to the bounds limits.

### `dominates`

```python
dominates(self, other: 'Agent') -> bool
```

Checks if this agent dominates another agent.

Returns:
    (bool): Whether this agent dominates the other.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `other` | `Agent` |  | Another agent to be compared. |

### `fill_with_binary`

```python
fill_with_binary(self) -> None
```

Fills the agent's decision variables with a binary distribution.

### `fill_with_static`

```python
fill_with_static(self, values: numpy.ndarray) -> None
```

Fills the agent's decision variables with static values. Note that this
method ignore the agent's bounds, so use it carefully.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `values` | `numpy.ndarray` |  | Values to be filled (expected shape (n_variables,)). |

### `fill_with_uniform`

```python
fill_with_uniform(self) -> None
```

Fills the agent's decision variables with a uniform distribution
based on bounds limits.
