---
title: Edge
description: API reference for Edge.
---

# `Edge`

**Module:** `opytimizer.core.graph.edge`

A connection between two `GraphNode`s.

## Constructor

```python
Edge(source: opytimizer.core.graph.node.GraphNode, target: opytimizer.core.graph.node.GraphNode, weight: Optional[float] = None, directed: bool = False) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | `opytimizer.core.graph.node.GraphNode` |  | Origin node. |
| `target` | `opytimizer.core.graph.node.GraphNode` |  | Destination node. |
| `weight` | `Optional[float]` | `None` | Optional numeric weight/cost associated with the edge. |
| `directed` | `bool` | `False` | Whether this edge should be treated as one-way (`source -> target` only) or two-way. |
