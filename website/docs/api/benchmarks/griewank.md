---
title: Griewank
description: API reference for Griewank.
---

# `Griewank`

**Module:** `opytimizer.benchmarks.single_objective`

Griewank: f(x) = 1 + sum(x_i^2) / 4000 - prod(cos(x_i / sqrt(i))). f* = 0 at 0.

## Constructor

```python
Griewank(n_variables: int = 30, lower_bound: Union[float, int, Sequence[float], NoneType] = None, upper_bound: Union[float, int, Sequence[float], NoneType] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` | `30` | Number of decision variables. |
| `lower_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Lower bound(s). Defaults to the standard domain of the problem. |
| `upper_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Upper bound(s). Defaults to the standard domain of the problem. |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
