---
title: ZCAT6
description: API reference for ZCAT6.
---

# `ZCAT6`

**Module:** `opytimizer.benchmarks.multi_objective.zcat`

ZCAT6: front with a logistic last objective.

## Constructor

```python
ZCAT6(n_variables: int = 30, n_objectives: int = 2, level: int = 1, bias: bool = False, imbalance: bool = False, complicated_pareto_set: bool = False, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` | `30` | Number of decision variables (at least n_objectives - 1, or 1 for ZCAT14-16). |
| `n_objectives` | `int` | `2` | Number of objectives (at least 2). |
| `level` | `int` | `1` | Difficulty of the distance landscape, from 1 (unimodal) to 6. |
| `bias` | `bool` | `False` | Whether a bias towards the Pareto set is introduced. |
| `imbalance` | `bool` | `False` | Whether odd and even objectives use different landscapes. |
| `complicated_pareto_set` | `bool` | `False` | Whether the Pareto set is curved (g_1..g_10) instead of a constant (g_0). |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
