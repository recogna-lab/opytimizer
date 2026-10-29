---
title: Function
description: API reference for Function.
---

# `Function`

**Module:** `opytimizer.core.function`

A Function class used to hold single-objective or multi-objective functions.

## Constructor

```python
Function(pointer: <built-in function callable>, budget: int = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `pointer` | `<built-in function callable>` |  | Pointer to a function or list of functions that will return the fitness value(s). |
| `budget` | `int` | `None` | — |
