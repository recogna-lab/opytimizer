---
title: Primitive
description: API reference for Primitive.
---

# `Primitive`

**Module:** `opytimizer.core.graph.primitive`

An internal (function) node: applies `function` over `arity` typed
children and produces a value of `output_type`.

## Constructor

```python
Primitive(name: str, function: Callable[..., Any], input_types: Tuple[Type, ...], output_type: Type) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` |  | Display name. |
| `function` | `Callable[..., Any]` |  | Callable applied to the evaluated children. |
| `input_types` | `Tuple[Type, ...]` |  | Expected type of each child, in order. |
| `output_type` | `Type` |  | Type produced by `function`. |
