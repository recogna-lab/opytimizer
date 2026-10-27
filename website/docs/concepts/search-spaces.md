---
sidebar_position: 5
title: Search Spaces
---

# Search Spaces

Opytimizer supports different topology representations for optimization tasks:

* **`SearchSpace`**: Standard $n$-dimensional real-valued vector space.
* **`BooleanSpace`**: Binary vector space, where each decision variable assumes a value in $\{0, 1\}$.
* **`GridSpace`**: Deterministic grid built from a step size per variable, where each grid point becomes an agent.
* **`HyperComplexSpace`**: Extends candidate positions to multi-dimensional hyper-complex numbers (e.g., Quaternions, Octonions).
* **`TreeSpace`**: Hierarchical tree topologies used primarily for Symbolic Regression and Genetic Programming (GP).
* **`GraphSpace`**: Graph-structured search topologies for network and pathing problems.

## Space Factories

Every space is a factory class that returns the appropriate implementation according to its arguments:

* **`n_objectives == 1`**: Builds a single-objective space, which tracks a single `best_agent`.
* **`n_objectives > 1`**: Builds a multi-objective space, which tracks a `pareto_front`.
* **`tensorized=True`**: Builds a tensorized space, where the whole population is stored in a single array and can be processed by tensorized optimizers.

## Bounds and Clipping

After each update, agents are clipped to the space bounds:

* $x_L \le x \le x_U$

The `lower_bound` and `upper_bound` arguments must have the same size as `n_variables`.
