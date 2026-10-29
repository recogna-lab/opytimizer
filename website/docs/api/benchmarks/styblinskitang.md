---
title: StyblinskiTang
description: API reference for StyblinskiTang.
---

# `StyblinskiTang`

**Module:** `opytimizer.benchmarks.single_objective`

Styblinski-Tang: f(x) = 0.5 sum(x_i^4 - 16 x_i^2 + 5 x_i).

f* = -39.16616570377142 n at x*_i = -2.903534027771178.

## Constructor

```python
StyblinskiTang(n_variables: int = 30, lower_bound: Union[float, int, Sequence[float], NoneType] = None, upper_bound: Union[float, int, Sequence[float], NoneType] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` | `30` | Number of decision variables. |
| `lower_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Lower bound(s). Defaults to the standard domain of the problem. |
| `upper_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Upper bound(s). Defaults to the standard domain of the problem. |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
