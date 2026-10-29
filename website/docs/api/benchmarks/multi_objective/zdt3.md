---
title: ZDT3
description: API reference for ZDT3.
---

# `ZDT3`

**Module:** `opytimizer.benchmarks.multi_objective.zdt`

ZDT3: front made of five disconnected convex parts.

## Constructor

```python
ZDT3(n_variables: Optional[int] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `Optional[int]` | `None` | Number of decision variables (at least 2). Defaults to the value recommended in the literature (30, or 10 for ZDT4 and ZDT6). |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
