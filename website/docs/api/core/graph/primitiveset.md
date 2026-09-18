---
title: PrimitiveSet
description: API reference for PrimitiveSet.
---

# `PrimitiveSet`

**Module:** `opytimizer.core.graph.primitive_set`

Registry of typed primitives, terminals and ephemeral constants.

## Constructor

```python
PrimitiveSet(name: str, root_type: Type) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` |  | Identifier for this set (useful when several problems/typed grammars coexist). |
| `root_type` | `Type` |  | The type that a tree generated from this set must produce at its root. |

## Methods

### `add_ephemeral_constant`

```python
add_ephemeral_constant(self, name: str, generator: Callable[[], Any], output_type: Type) -> None
```

Registers a leaf that is freshly sampled from `generator` every
time it's drawn during tree generation.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` |  | — |
| `generator` | `Callable[[], Any]` |  | — |
| `output_type` | `Type` |  | — |

### `add_primitive`

```python
add_primitive(self, function: Callable[..., Any], input_types: Tuple[Type, ...], output_type: Type, name: Optional[str] = None) -> None
```

Registers a typed function node.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `function` | `Callable[..., Any]` |  | Callable to be applied to the node's children. |
| `input_types` | `Tuple[Type, ...]` |  | Expected type of each child, in order. |
| `output_type` | `Type` |  | Type produced by `function`. |
| `name` | `Optional[str]` | `None` | Optional display name (defaults to `function.__name__`). |

### `add_terminal`

```python
add_terminal(self, value: Any, output_type: Type, name: Optional[str] = None) -> None
```

Registers a fixed leaf value.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `value` | `Any` |  | — |
| `output_type` | `Type` |  | — |
| `name` | `Optional[str]` | `None` | — |

### `has_terminal`

```python
has_terminal(self, output_type: Type) -> bool
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `output_type` | `Type` |  | — |

### `primitives_of`

```python
primitives_of(self, output_type: Type) -> List[opytimizer.core.graph.primitive.Primitive]
```

Returns every primitive whose `output_type` matches.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `output_type` | `Type` |  | — |

### `terminals_of`

```python
terminals_of(self, output_type: Type) -> List[Union[opytimizer.core.graph.primitive.Terminal, opytimizer.core.graph.primitive.Ephemeral]]
```

Returns every terminal/ephemeral whose `output_type` matches.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `output_type` | `Type` |  | — |

### `validate`

```python
validate(self) -> None
```

Walks every registered primitive's `input_types` and raises if
any of them has no matching terminal/primitive registered — i.e.,
the grammar would eventually get stuck trying to close a branch.

Call this once, right after registering everything.
