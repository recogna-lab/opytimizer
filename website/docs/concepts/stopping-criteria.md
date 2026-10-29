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

## Progress Bars

Each criterion draws its own progress bar during `start()`. They can be disabled with `show_progress=False` when creating the `Opytimizer`, which is convenient for long benchmark studies with many runs:

```python
opt = Opytimizer(space, optimizer, function, show_progress=False)
```

Without progress bars, the values shown in the bar (e.g., the best fitness) are never copied back from the GPU. On GPU runs, `NoImprovement(..., check_every=10)` also reduces how often the criterion reads its counter back to the host.

## Example Code

```python
from opytimizer.core import MaxEvaluations, MaxIterations, NoImprovement

# Single stopping condition
stopping_criterion = MaxIterations(100)

# Multiple criteria can be specified
# The optimization terminates if any criterion is met
stopping_criteria = [MaxEvaluations(1000), NoImprovement(patience=30, min_delta=1e-5)]
```
