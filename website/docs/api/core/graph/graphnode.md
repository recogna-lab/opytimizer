---
title: GraphNode
description: API reference for GraphNode.
---

# `GraphNode`

**Module:** `opytimizer.core.graph.node`

A single vertex of a `Graph`/`Tree`.

## Constructor

```python
GraphNode(name: str, value: Any = None, output_type: Optional[Type] = None, is_terminal: bool = True) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` |  | Display/identifier name (e.g., the primitive's name, or a variable's name). |
| `value` | `Any` | `None` | Payload — for a terminal, the actual value; for a function node, the callable that should be applied to the node's children's evaluated outputs. |
| `output_type` | `Optional[Type]` | `None` | Type produced when this node is evaluated. `None` means "untyped" (kept for plain graphs that don't need strongly-typed GP semantics). |
| `is_terminal` | `bool` | `True` | Whether this node is a leaf (terminal) or an internal function node. |

## Methods

### `add_child`

```python
add_child(self, child: 'GraphNode', expected_type: Optional[Type] = None) -> None
```

Attaches `child` to this node.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `child` | `GraphNode` |  | Node to attach. |
| `expected_type` | `Optional[Type]` | `None` | If provided (typed mode), validates that `child.output_type` matches before attaching. |

### `copy`

```python
copy(self) -> 'GraphNode'
```

Performs a deep, structure-preserving copy of the subtree/subgraph
rooted at this node (recursively copies children).

### `evaluate`

```python
evaluate(self) -> Any
```

Recursively evaluates this node: terminals return their value;
function nodes apply `self.value` (a callable) over the evaluated
children.

### `remove_child`

```python
remove_child(self, child: 'GraphNode') -> None
```

Detaches `child` from this node (and vice-versa).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `child` | `GraphNode` |  | — |
