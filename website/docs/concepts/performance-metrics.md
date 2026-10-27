---
sidebar_position: 12
title: Performance Metrics
---

# Performance Metrics

Multi-objective algorithms return a set of solutions instead of a single best agent. Performance metrics quantify the convergence and the diversity of the obtained Pareto front.

## Convergence Metrics

* **`GD`**: Generational Distance, the average distance from each obtained solution to the true Pareto front.
* **`IGD`**: Inverted Generational Distance, the average distance from each point of the true Pareto front to the obtained solutions.
* **`ErrorRatio`**: Fraction of obtained solutions that do not belong to the true Pareto front (within a tolerance `tol`).

## Diversity Metrics

* **`Spread`**: Measures how uniformly the obtained solutions are distributed along the front.
* **`MaximumSpread`**: Largest distance between any two solutions of the obtained front.

## Combined Metrics

* **`HV`**: Hypervolume, the volume of the objective space dominated by the obtained front and bounded by a `reference_point`. Higher values are better.
* **`MonteCarloHV`**: Hypervolume approximated by Monte Carlo sampling, suited for many-objective problems.
* **`R2`**: Utility-based indicator computed over a set of weight vectors.

## Formulation

Given an obtained front $A$ and a reference front $P^*$, the IGD is computed as:

* $IGD(A, P^*) = \frac{1}{|P^*|} \sum_{v \in P^*} \min_{a \in A} d(v, a)$

Where $d(\cdot, \cdot)$ is the Euclidean distance.

## Usage

Metrics are passed to the `start()` method through the `metrics` argument. Their values are displayed in the progress bar and, when `save_history=True`, stored in the history under the lowercase metric name (e.g., `opt.history.hv`).

Metrics that require a reference front (`GD`, `IGD`, `Spread`, `ErrorRatio`) receive it as their first argument.

Since computing the Pareto front and the metrics can be expensive, the `metrics_every` argument of `start()` computes them only every `metrics_every` iterations (default `1`). The final front is always computed at the end of the task.
