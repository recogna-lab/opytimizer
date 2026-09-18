---
title: GraphAgent
description: API reference for GraphAgent.
---

# `GraphAgent`

**Module:** `opytimizer.core.graph.agent_graph`

An Agent whose position is a `Graph` (or `Tree`, since `Tree`
subclasses `Graph`) instance instead of a numeric ndarray.

## Constructor

```python
GraphAgent(n_objectives: int = 1, mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_objectives` | `int` | `1` | Number of objective functions. |
| `mapping` | `Optional[List[str]]` | `None` | Optional display name(s) for this agent's payload. |
| `env` | `opytimizer.core.environment.Environment` | `None` | Environment class object. |

## Methods

### `dominates`

```python
dominates(self, other: 'GraphAgent') -> bool
```

Checks if this agent dominates another agent.

Returns:
    (bool): Whether this agent dominates the other.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `other` | `GraphAgent` |  | Another agent to be compared. |
