---
title: Graph
description: API reference for Graph.
---

# `Graph`

**Module:** `opytimizer.core.graph.graph`

Holds a collection of `GraphNode`s and `Edge`s.

## Constructor

```python
Graph(directed: bool = False) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `directed` | `bool` | `False` | Whether edges should be interpreted as one-way. |

## Methods

### `add_edge`

```python
add_edge(self, source: opytimizer.core.graph.node.GraphNode, target: opytimizer.core.graph.node.GraphNode, weight: Optional[float] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | `opytimizer.core.graph.node.GraphNode` |  | — |
| `target` | `opytimizer.core.graph.node.GraphNode` |  | — |
| `weight` | `Optional[float]` | `None` | — |

### `add_node`

```python
add_node(self, node: opytimizer.core.graph.node.GraphNode) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `node` | `opytimizer.core.graph.node.GraphNode` |  | — |

### `copy`

```python
copy(self) -> 'Graph'
```

Deep-copies the graph, preserving node identity via a mapping
table (so shared references — e.g., a DAG's converging edges —
survive the copy).

### `neighbors`

```python
neighbors(self, node: opytimizer.core.graph.node.GraphNode) -> List[opytimizer.core.graph.node.GraphNode]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `node` | `opytimizer.core.graph.node.GraphNode` |  | — |

### `remove_edge`

```python
remove_edge(self, edge: opytimizer.core.graph.edge.Edge) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `edge` | `opytimizer.core.graph.edge.Edge` |  | — |
