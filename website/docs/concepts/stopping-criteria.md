---
sidebar_position: 8
title: Stopping Criteria
---

# Stopping Criteria

Stopping criteria define when the optimization loop should end. They are passed to the `start()` method of `Opytimizer`.

## Available Criteria

* **`MaxIterations(n)`**: Stops after `n` iterations.
* **`MaxEvaluations(n)`**: Stops once the objective function has been called at least `n` times. It is verified at the end of each iteration.
* **`NoImprovement(patience, min_delta)`**: Stops when the best fitness does not improve by more than `min_delta` for `patience` consecutive iterations. It is only available for single-objective tasks.

## Combining Criteria

A list of criteria may be provided, and the optimization terminates as soon as **any** criterion is met.

If no criterion is provided, `MaxIterations(250)` is used by default.

Note that `MaxEvaluations` is a soft check: since it is only verified at the end of each iteration, the final number of evaluations may slightly exceed `n`.

## Example Code

```python
from opytimizer.core import MaxEvaluations, MaxIterations, NoImprovement

# Single stopping condition
stopping_criterion = MaxIterations(100)

# Multiple criteria can be specified
# The optimization terminates if any criterion is met
stopping_criteria = [MaxEvaluations(1000), NoImprovement(patience=30, min_delta=1e-5)]
```
