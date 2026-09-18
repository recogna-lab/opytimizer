---
title: Tree
description: API reference for Tree.
---

# `Tree`

**Module:** `opytimizer.core.graph.tree`

A rooted tree: a directed, acyclic `Graph` with a single root and at
most one parent per node.

## Constructor

```python
Tree(root: Optional[opytimizer.core.graph.node.GraphNode] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `root` | `Optional[opytimizer.core.graph.node.GraphNode]` | `None` | The tree's root node. May be `None` for an empty tree that will be populated later (e.g., by a generator). |

## Methods

### `copy`

```python
copy(self) -> 'Tree'
```

Deep-copies the graph, preserving node identity via a mapping
table (so shared references — e.g., a DAG's converging edges —
survive the copy).

### `evaluate`

```python
evaluate(self)
```

Evaluates the whole tree by recursively evaluating its root.

### `nodes_of_type`

```python
nodes_of_type(self, output_type: Type) -> List[opytimizer.core.graph.node.GraphNode]
```

Returns every node whose `output_type` matches — used by
type-safe crossover/mutation to pick compatible cut points.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `output_type` | `Type` |  | — |

### `replace_subtree`

```python
replace_subtree(self, old_node: opytimizer.core.graph.node.GraphNode, new_node: opytimizer.core.graph.node.GraphNode) -> None
```

Replaces `old_node` (and its whole subtree) with `new_node`
in place, updating parent links and the cached node list.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `old_node` | `opytimizer.core.graph.node.GraphNode` |  | — |
| `new_node` | `opytimizer.core.graph.node.GraphNode` |  | — |
