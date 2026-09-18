---
title: GaussianMutationTensor
description: API reference for GaussianMutationTensor.
---

# `GaussianMutationTensor`

**Module:** `opytimizer.utils.operators`

Fully vectorized Gaussian Mutation running directly on GPU or Tensor CPU arrays.

## Constructor

```python
GaussianMutationTensor(env: opytimizer.core.environment.Environment, rate: float = 0.025, std: float = 0.1)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `env` | `opytimizer.core.environment.Environment` |  | — |
| `rate` | `float` | `0.025` | — |
| `std` | `float` | `0.1` | — |
