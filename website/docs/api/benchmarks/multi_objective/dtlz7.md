---
title: DTLZ7
description: API reference for DTLZ7.
---

# `DTLZ7`

**Module:** `opytimizer.benchmarks.multi_objective.dtlz`

DTLZ7: front with 2^(M-1) disconnected regions.

## Constructor

```python
DTLZ7(n_variables: Optional[int] = None, n_objectives: int = 3, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `Optional[int]` | `None` | Number of decision variables (at least `n_objectives`). Defaults to n_objectives + k - 1, with the k recommended in the literature (5 for DTLZ1, 10 for DTLZ2-6 and 20 for DTLZ7). |
| `n_objectives` | `int` | `3` | Number of objectives (at least 2). |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
