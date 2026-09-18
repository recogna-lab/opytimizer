---
sidebar_position: 1
title: "Optimization Fundamentals"
slug: /
---

# Optimization Fundamentals

Optimization is the field of applied mathematics dedicated to finding the best solution from a set of available alternatives according to specified criteria.

## Mathematical Formulation

A general single-objective optimization problem can be formulated as:

* **Objective Function**: Minimize $f(x)$
* **Decision Variables**: $x = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n$
* **Search Bounds**: $x_L \le x \le x_U$

Where:
* $x$ is the decision vector in $n$-dimensional space.
* $f(x)$ is the objective (or fitness) function.
* $x_L$ and $x_U$ define the lower and upper search bounds.

## Meta-Heuristics

In complex non-convex search spaces, traditional derivative-based methods (such as Gradient Descent) often get trapped in local optima or fail due to non-differentiability. 

Meta-heuristics provide stochastic optimization techniques capable of exploring large spaces efficiently without requiring gradient information.