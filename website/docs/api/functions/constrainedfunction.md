---
title: ConstrainedFunction
description: API reference for ConstrainedFunction.
---

# `ConstrainedFunction`

**Module:** `opytimizer.functions.constrained`

A ConstrainedFunction class used to hold constrained single-objective functions.

## Constructor

```python
ConstrainedFunction(pointer: List[<built-in function callable>], constraints: List[<built-in function callable>], penalty: float = 0.0) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `pointer` | `List[<built-in function callable>]` |  | Pointer to a function that will return the fitness value. |
| `constraints` | `List[<built-in function callable>]` |  | Constraints to be applied to the fitness function. |
| `penalty` | `float` | `0.0` | Penalization factor when a constraint is not valid. |
