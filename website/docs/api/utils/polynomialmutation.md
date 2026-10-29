---
title: PolynomialMutation
description: API reference for PolynomialMutation.
---

# `PolynomialMutation`

**Module:** `opytimizer.utils.operators`

Polynomial mutation for real-valued vectors operating on Agents (CPU).

Follows the bounded polynomial mutation of Deb's NSGA-II reference implementation.

## Constructor

```python
PolynomialMutation(eta: int = 20, rate: float = 0.03333333333333333)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `eta` | `int` | `20` | Distribution index (larger values yield smaller perturbations). |
| `rate` | `float` | `0.03333333333333333` | Probability of mutating each variable (usually 1 / n_variables). |
