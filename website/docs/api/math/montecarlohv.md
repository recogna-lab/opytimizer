---
title: MonteCarloHV
description: API reference for MonteCarloHV.
---

# `MonteCarloHV`

**Module:** `opytimizer.math.metrics`

Hypervolume estimated by Monte Carlo sampling (suited for many objectives).

Samples are drawn uniformly from the box [lower_bound, upper_bound], and the
hypervolume is the fraction of samples dominated by the front times the
volume of the box. `upper_bound` plays the role of the reference point and
`lower_bound` should be no worse than the ideal point of the front.

## Constructor

```python
MonteCarloHV(n_samples: int = 1000000, lower_bound: Union[float, List[float]] = 0.0, upper_bound: Union[float, List[float]] = 1.0, parallel: bool = False, diff: float = None)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_samples` | `int` | `1000000` | Number of Monte Carlo samples. |
| `lower_bound` | `Union[float, List[float]]` | `0.0` | Lower corner of the sampling box (scalar or per objective). |
| `upper_bound` | `Union[float, List[float]]` | `1.0` | Upper corner of the sampling box, i.e., the reference point. |
| `parallel` | `bool` | `False` | Whether the dominance checks are split among threads. |
| `diff` | `float` | `None` | Deprecated and ignored. It used to enlarge the measured volume without enlarging the sampled box, which biased the estimate. |
