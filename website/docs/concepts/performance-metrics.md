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

* **`Spread`**: Deb's $\Delta$, which measures how uniformly the obtained solutions are distributed along the front and how far its boundary solutions are from the extreme solutions of the true front. A perfectly spread front that reaches both extremes has $\Delta = 0$.
* **`MaximumSpread`**: Diagonal of the bounding box of the obtained front, $MS = \sqrt{\sum_{m=1}^{M} (\max f_m - \min f_m)^2}$ (Zitzler, Deb and Thiele, 2000).

## Combined Metrics

* **`HV`**: Hypervolume, the volume of the objective space dominated by the obtained front and bounded by a `reference_point`. Higher values are better. It is computed exactly for any number of objectives.
* **`MonteCarloHV`**: Hypervolume estimated by Monte Carlo sampling of the box `[lower_bound, upper_bound]`, where `upper_bound` plays the role of the reference point. Its cost does not grow with the number of objectives, so it is suited for many-objective problems.
* **`R2`**: Utility-based indicator computed over a set of weight vectors.

## Formulation

Given an obtained front $A$ and a reference front $P^*$, the IGD is computed as:

* $IGD(A, P^*) = \frac{1}{|P^*|} \sum_{v \in P^*} \min_{a \in A} d(v, a)$

Where $d(\cdot, \cdot)$ is the Euclidean distance. `GD`, `IGD`, `Spread` and `ErrorRatio` normalize both fronts by the range of the reference front before measuring distances.

The hypervolume of a front $A$ with respect to a reference point $r$ is:

* $HV(A, r) = \lambda\left(\bigcup_{a \in A} [a_1, r_1] \times \dots \times [a_M, r_M]\right)$
ight)$

Where $\lambda$ is the Lebesgue measure. Only points strictly better than $r$ in every objective contribute to it.

## Hypervolume Algorithms

`HV` is exact and chooses its algorithm by the number of objectives $M$:

| Objectives | Algorithm | Cost |
| --- | --- | --- |
| $M = 1, 2$ | Sweep over the sorted front | $O(n \log n)$ |
| $M = 3$ | Slicing along $f_3$, with a 2-D sweep per slice | $O(n^2 \log n)$ |
| $M \geq 4$ | WFG algorithm (While, Bradstreet and Barone, 2012), slicing the last objective down to the 3-D case | Exponential in $M$ in the worst case |

The exact hypervolume becomes expensive as $M$ grows: as a reference, a front of 100 points takes tens of milliseconds with four objectives and a few hundred milliseconds with five. For many objectives or large fronts, `MonteCarloHV` gives an unbiased estimate whose cost depends only on `n_samples`.

```python
from opytimizer.math.metrics import HV, MonteCarloHV

# Exact, for any number of objectives
hv = HV(reference_point=[1.1, 1.1, 1.1, 1.1, 1.1])

# Estimated, with the reference point as the upper corner of the sampled box
mc_hv = MonteCarloHV(n_samples=10**6, lower_bound=0.0, upper_bound=1.1)
```

Scalar bounds of `MonteCarloHV` are expanded to the number of objectives of the front.

## Usage

Metrics are passed to the `start()` method through the `metrics` argument. Their values are displayed in the progress bar and, when `save_history=True`, stored in the history under the lowercase metric name (e.g., `opt.history.hv`).

Metrics that require a reference front (`GD`, `IGD`, `Spread`, `ErrorRatio`) receive it as their first argument. The true front of a benchmark problem, `problem.pareto_front()`, can be used directly (see [Benchmark Problems](benchmarks.md)).

## Backends

Metrics run independently of the `Environment` of the task. They accept:

* NumPy or CuPy arrays of shape `(n_points, n_objectives)`;
* the `(positions, fitness)` tuple of a tensorized space's Pareto front;
* a list of agents (standard spaces), whose fitness may live on either backend.

Reference fronts, reference points, weight vectors and bounds may also be CuPy arrays. Every input is copied to the host and the metric is computed with NumPy: fronts are small, so a single device-to-host copy per evaluation is negligible next to the optimization itself, and the returned value is always a host float. CuPy is never imported by the metrics module, so it works on machines without it.

Since computing the Pareto front and the metrics can be expensive, the `metrics_every` argument of `start()` computes them only every `metrics_every` iterations (default `1`). The final front is always computed at the end of the task.
