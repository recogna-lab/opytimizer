---
title: DixonPrice
description: API reference for DixonPrice.
---

# `DixonPrice`

**Module:** `opytimizer.benchmarks.single_objective`

Dixon-Price: f(x) = (x_1 - 1)^2 + sum_\{i>1\} i (2 x_i^2 - x_\{i-1\})^2. f* = 0.

## Constructor

```python
DixonPrice(n_variables: int = 30, lower_bound: Union[float, int, Sequence[float], NoneType] = None, upper_bound: Union[float, int, Sequence[float], NoneType] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` | `30` | Number of decision variables. |
| `lower_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Lower bound(s). Defaults to the standard domain of the problem. |
| `upper_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Upper bound(s). Defaults to the standard domain of the problem. |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
