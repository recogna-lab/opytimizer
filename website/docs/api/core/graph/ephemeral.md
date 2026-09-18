---
title: Ephemeral
description: API reference for Ephemeral.
---

# `Ephemeral`

**Module:** `opytimizer.core.graph.primitive`

A leaf whose value is (re)sampled from `generator` every time it is
drawn during tree generation — e.g., a random constant.

## Constructor

```python
Ephemeral(name: str, generator: Callable[[], Any], output_type: Type) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` |  | Display name. |
| `generator` | `Callable[[], Any]` |  | Zero-argument callable producing a fresh value. |
| `output_type` | `Type` |  | Type of the values `generator` produces. |
