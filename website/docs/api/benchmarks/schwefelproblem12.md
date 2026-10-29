---
title: SchwefelProblem12
description: API reference for SchwefelProblem12.
---

# `SchwefelProblem12`

**Module:** `opytimizer.benchmarks.single_objective`

Schwefel's problem 1.2: f(x) = sum_i (sum_\{j\<=i\} x_j)^2. f* = 0 at x* = 0.

## Constructor

```python
SchwefelProblem12(n_variables: int = 30, lower_bound: Union[float, int, Sequence[float], NoneType] = None, upper_bound: Union[float, int, Sequence[float], NoneType] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` | `30` | Number of decision variables. |
| `lower_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Lower bound(s). Defaults to the standard domain of the problem. |
| `upper_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Upper bound(s). Defaults to the standard domain of the problem. |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
