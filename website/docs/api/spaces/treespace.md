---
title: TreeSpace
description: API reference for TreeSpace.
---

# `TreeSpace`

**Module:** `opytimizer.spaces.tree`

A TreeSpace Factory Class for agents, variables and methods related
to strongly-typed genetic programming trees.

Note:
    A `tensorized` variant is a natural future extension point but is not implemented
    yet, since tree topologies are inherently variable-sized.
    Passing `tensorized=True` raises `NotImplementedError` rather than
    failing silently.

## Constructor

```python
TreeSpace(n_agents: int, n_objectives: int, pset: opytimizer.core.graph.primitive_set.PrimitiveSet, min_depth: int = 2, max_depth: int = 6, method: str = 'half_and_half', mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None, tensorized: bool = False)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_agents` | `int` |  | — |
| `n_objectives` | `int` |  | — |
| `pset` | `opytimizer.core.graph.primitive_set.PrimitiveSet` |  | — |
| `min_depth` | `int` | `2` | — |
| `max_depth` | `int` | `6` | — |
| `method` | `str` | `'half_and_half'` | — |
| `mapping` | `Optional[List[str]]` | `None` | — |
| `env` | `opytimizer.core.environment.Environment` | `None` | — |
| `tensorized` | `bool` | `False` | — |
