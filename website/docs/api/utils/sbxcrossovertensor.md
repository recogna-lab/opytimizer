---
title: SBXCrossoverTensor
description: API reference for SBXCrossoverTensor.
---

# `SBXCrossoverTensor`

**Module:** `opytimizer.utils.operators`

Simulated Binary Crossover (SBX) re-architected for tensorized operations.

## Constructor

```python
SBXCrossoverTensor(env: opytimizer.core.environment.Environment, eta: int = 20, rate: float = 1.0, gene_rate: float = 0.5)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `env` | `opytimizer.core.environment.Environment` |  | Environment whose array module (numpy / cupy) runs the operator. |
| `eta` | `int` | `20` | Distribution index (larger values keep children closer to parents). |
| `rate` | `float` | `1.0` | Probability of applying the crossover to a pair of parents. |
| `gene_rate` | `float` | `0.5` | Probability of recombining each variable of a pair. The standard value (0.5) preserves building blocks; recombining every variable (1.0) severely slows down convergence. |
