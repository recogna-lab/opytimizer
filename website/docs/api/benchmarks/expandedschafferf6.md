---
title: ExpandedSchafferF6
description: API reference for ExpandedSchafferF6.
---

# `ExpandedSchafferF6`

**Module:** `opytimizer.benchmarks.single_objective`

Expanded Schaffer's F6 (CEC base function), summed over consecutive pairs.

f* = 0 at x* = 0.

## Constructor

```python
ExpandedSchafferF6(n_variables: int = 30, lower_bound: Union[float, int, Sequence[float], NoneType] = None, upper_bound: Union[float, int, Sequence[float], NoneType] = None, env: Optional[opytimizer.core.environment.Environment] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` | `30` | Number of decision variables. |
| `lower_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Lower bound(s). Defaults to the standard domain of the problem. |
| `upper_bound` | `Union[float, int, Sequence[float], NoneType]` | `None` | Upper bound(s). Defaults to the standard domain of the problem. |
| `env` | `Optional[opytimizer.core.environment.Environment]` | `None` | Environment that runs the problem. Defaults to numpy / float64. |
