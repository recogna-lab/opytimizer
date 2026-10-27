---
sidebar_position: 7
title: Objective Functions
---

# Objective Functions

Objective functions measure the quality of a candidate solution. Opytimizer always minimizes the value returned by the function.

## Function Classes

* **`Function`**: Wraps a callable (or a list of callables) that receives an agent's position and returns its fitness value(s).
* **`ConstrainedFunction`**: Wraps a callable and a list of constraints, penalizing positions that violate them.
* **`WeightedFunction`**: Combines several callables into a single objective through a weighted sum.

## Single and Multiple Objectives

When a list of callables is passed to `Function`, it returns an objective vector:

* $F(x) = [f_1(x), f_2(x), \dots, f_M(x)]^T$

Alternatively, a single callable may return the whole objective vector at once.

## Constraints

Each constraint is a callable that returns `True` when the position is feasible. Infeasible positions receive a penalty proportional to their fitness:

* $f_p(x) = f(x) + \lambda \cdot f(x)$, for every violated constraint

Where $\lambda$ is the `penalty` factor.

## Weighted Aggregation

`WeightedFunction` reduces $M$ objectives to a single scalar value:

* $f(x) = \sum_{i=1}^{M} w_i f_i(x)$

## Evaluation Counter

Every `Function` counts its calls in `n_calls`, which is used by the `MaxEvaluations` stopping criterion.

## Maximization

To maximize a function, simply return its negative value:

* $\max f(x) = -\min (-f(x))$
