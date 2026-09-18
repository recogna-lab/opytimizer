---
title: GraphSpace
description: API reference for GraphSpace.
---

# `GraphSpace`

**Module:** `opytimizer.spaces.graph`

A GraphSpace Factory Class for agents, variables and methods related
to graph-structured search spaces (directed/undirected,
tensorized/non-tensorized).

## Constructor

```python
GraphSpace(n_agents: int, n_nodes: int, n_objectives: int, directed: bool = False, edge_prob: float = 0.3, connected: bool = False, mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None, tensorized: bool = False) -> Union[opytimizer.spaces.graph._SingleObjectiveGraphSpace, opytimizer.spaces.graph._MultiObjectiveGraphSpace, opytimizer.spaces.graph._SingleObjectiveTensorGraphSpace, opytimizer.spaces.graph._MultiObjectiveTensorGraphSpace]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_agents` | `int` |  | — |
| `n_nodes` | `int` |  | — |
| `n_objectives` | `int` |  | — |
| `directed` | `bool` | `False` | — |
| `edge_prob` | `float` | `0.3` | — |
| `connected` | `bool` | `False` | — |
| `mapping` | `Optional[List[str]]` | `None` | — |
| `env` | `opytimizer.core.environment.Environment` | `None` | — |
| `tensorized` | `bool` | `False` | — |
