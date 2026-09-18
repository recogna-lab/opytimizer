---
title: MonteCarloHV
description: API reference for MonteCarloHV.
---

# `MonteCarloHV`

**Module:** `opytimizer.math.metrics`

Helper class that provides a standard way to create an ABC using
inheritance.

## Constructor

```python
MonteCarloHV(n_samples: int = 1000000, lower_bound: Union[float, List[float]] = 0.0, upper_bound: Union[float, List[float]] = 1.0, parallel: bool = False, diff: float = 0.01)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_samples` | `int` | `1000000` | — |
| `lower_bound` | `Union[float, List[float]]` | `0.0` | — |
| `upper_bound` | `Union[float, List[float]]` | `1.0` | — |
| `parallel` | `bool` | `False` | — |
| `diff` | `float` | `0.01` | — |
