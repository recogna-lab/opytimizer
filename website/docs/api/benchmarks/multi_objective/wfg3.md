---
title: WFG3
description: API reference for WFG3.
---

# `WFG3`

**Module:** `opytimizer.benchmarks.multi_objective.wfg`

WFG3: linear and degenerate front (a line), non-separable distance.

## Constructor

```python
WFG3(n_variables: Optional[int] = None, n_objectives: int = 3, k: Optional[int] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `Optional[int]` | `None` | Number of decision variables (k + l). Defaults to k + 20. |
| `n_objectives` | `int` | `3` | Number of objectives (at least 2). |
| `k` | `Optional[int]` | `None` | Number of position-related variables, a multiple of n_objectives - 1. Defaults to 4 for two objectives and to 2 (n_objectives - 1) otherwise. |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
