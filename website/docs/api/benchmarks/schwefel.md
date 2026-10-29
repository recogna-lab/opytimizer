---
title: Schwefel
description: API reference for Schwefel.
---

# `Schwefel`

**Module:** `opytimizer.benchmarks.single_objective`

Schwefel's problem 2.26: f(x) = 418.9829 n - sum(x_i sin(sqrt|x_i|)).

Deceptive multimodal function; f* ~ 0 at x* ~ 420.9687.

## Constructor

```python
Schwefel(n_variables: int = 30, lower_bound: Union[float, int, Sequence[float], NoneType] = None, upper_bound: Union[float, int, Sequence[float], NoneType] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` | `30` | Number of decision variables. |
| `lower_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Lower bound(s). Defaults to the standard domain of the problem. |
| `upper_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Upper bound(s). Defaults to the standard domain of the problem. |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
