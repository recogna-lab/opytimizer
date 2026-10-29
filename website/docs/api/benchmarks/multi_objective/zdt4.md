---
title: ZDT4
description: API reference for ZDT4.
---

# `ZDT4`

**Module:** `opytimizer.benchmarks.multi_objective.zdt`

ZDT4: convex front with 21^9 local fronts (Rastrigin-like g).

## Constructor

```python
ZDT4(n_variables: Optional[int] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `Optional[int]` | `None` | Number of decision variables (at least 2). Defaults to the value recommended in the literature (30, or 10 for ZDT4 and ZDT6). |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
