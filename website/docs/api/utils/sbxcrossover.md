---
title: SBXCrossover
description: API reference for SBXCrossover.
---

# `SBXCrossover`

**Module:** `opytimizer.utils.operators`

Simulated Binary Crossover (SBX) operating on Agents (CPU).

Follows the bounded SBX of Deb's NSGA-II reference implementation.

References:
    K. Deb and R. B. Agrawal, "Simulated binary crossover for continuous search
    space," Complex Systems, vol. 9, no. 2, pp. 115-148, 1995.

## Constructor

```python
SBXCrossover(eta: int = 20, rate: float = 1.0, gene_rate: float = 0.5, n_offspring: int = 2)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `eta` | `int` | `20` | Distribution index (larger values keep children closer to parents). |
| `rate` | `float` | `1.0` | Probability of applying the crossover to a pair of parents. |
| `gene_rate` | `float` | `0.5` | Probability of recombining each variable of a pair. The standard value (0.5) preserves building blocks; recombining every variable (1.0) severely slows down convergence. |
| `n_offspring` | `int` | `2` | Number of children returned by a single-pair call. |
