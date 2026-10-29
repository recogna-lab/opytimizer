---
sidebar_position: 7.5
title: Benchmark Problems
---

# Benchmark Problems

The `opytimizer.benchmarks` package provides test problems from the literature, ready to be plugged into any optimizer. It covers classic single-objective functions, the CEC 2022 competition suite and four scalable multi- and many-objective families: ZDT, DTLZ, WFG and ZCAT.

```python
from opytimizer.benchmarks import Rastrigin, CEC2022F4, ZDT1, DTLZ2, WFG4, ZCAT3
```

## Design

Every benchmark follows the same contract:

* **It is a callable.** It can be wrapped directly by `Function`, `ConstrainedFunction` or `WeightedFunction`, just like a user-defined function.
* **It is configured by the user.** The number of variables (and of objectives, for scalable families) is chosen at construction time. Defaults follow the values recommended by the original papers.
* **It runs on the backend of its `env`.** Passing `env=Environment("cupy", ...)` evaluates the problem on the GPU. Without an `env`, it runs on NumPy with `float64`. CuPy is only imported when a CuPy environment is actually used, so the package works on machines without it.
* **It exposes its search domain.** `lower_bound` and `upper_bound` are lists that can be passed straight to `SearchSpace`.
* **It knows its optimum.** Single-objective problems expose `optimal_value` and `optimal_solution()`; multi-objective problems expose `pareto_front()`.

```python
from opytimizer.benchmarks import DTLZ2
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace

problem = DTLZ2(n_variables=12, n_objectives=3)

space = SearchSpace(
    100,
    problem.n_variables,
    problem.n_objectives,
    problem.lower_bound,
    problem.upper_bound,
)
function = Function(problem)
```

### Accepted Inputs

A benchmark recognizes both evaluation modes of Opytimizer, so the same object works with standard and tensorized optimizers:

| Input shape | Origin | Output (single-objective) | Output (multi-objective) |
| --- | --- | --- | --- |
| `(n_variables,)` | Direct call | scalar | `(n_objectives,)` |
| `(n_variables, 1)` | Position of an `Agent` (standard optimizers) | scalar | `(n_objectives,)` |
| `(N, n_variables)` | Batch of solutions | `(N,)` | `(N, n_objectives)` |
| `(N, n_variables, n_dimensions)` | Tensorized space (`*Tensor`, `*Cuda` optimizers) | `(N,)` | `(N, n_objectives)` |

Inputs are cast to the dtype of the environment and moved to its device when needed. Outputs always live on the environment's backend.

:::tip
Agent-by-agent optimizers (the standard, non-tensorized ones) should use a NumPy environment: evaluating one solution at a time on the GPU would copy data between devices at every call and is not supported by `Function` in that mode.
:::

## Single-Objective Functions

Classic n-dimensional functions in their standard (non-shifted) form. Each one accepts `n_variables` (default `30`) and optional `lower_bound` / `upper_bound`, which replace the standard domain below.

| Class | Domain | $f^*$ | $x^*$ | Features |
| --- | --- | --- | --- | --- |
| `Sphere` | $[-100, 100]$ | 0 | $0$ | Unimodal, separable |
| `SumSquares` | $[-10, 10]$ | 0 | $0$ | Unimodal, separable |
| `HighConditionedElliptic` | $[-100, 100]$ | 0 | $0$ | Unimodal, ill-conditioned |
| `BentCigar` | $[-100, 100]$ | 0 | $0$ | Unimodal, ill-conditioned |
| `Discus` | $[-100, 100]$ | 0 | $0$ | Unimodal, ill-conditioned |
| `Quartic` | $[-1.28, 1.28]$ | 0 | $0$ | Unimodal (noiseless) |
| `Step` | $[-100, 100]$ | 0 | $[-0.5, 0.5)^n$ | Discontinuous plateaus |
| `SchwefelProblem12` | $[-100, 100]$ | 0 | $0$ | Unimodal, non-separable |
| `SchwefelProblem222` | $[-10, 10]$ | 0 | $0$ | Unimodal |
| `Zakharov` | $[-5, 10]$ | 0 | $0$ | Unimodal, non-separable |
| `Rosenbrock` | $[-5, 10]$ | 0 | $1$ | Narrow curved valley |
| `DixonPrice` | $[-10, 10]$ | 0 | $2^{-(2^i-2)/2^i}$ | Valley, non-separable |
| `Rastrigin` | $[-5.12, 5.12]$ | 0 | $0$ | Highly multimodal |
| `Ackley` | $[-32.768, 32.768]$ | 0 | $0$ | Multimodal, flat outer region |
| `Griewank` | $[-600, 600]$ | 0 | $0$ | Multimodal, non-separable |
| `Schwefel` | $[-500, 500]$ | 0 | $420.9687$ | Deceptive, multimodal |
| `Levy` | $[-10, 10]$ | 0 | $1$ | Multimodal |
| `StyblinskiTang` | $[-5, 5]$ | $-39.16617\,n$ | $-2.903534$ | Multimodal |
| `Alpine` | $[-10, 10]$ | 0 | $0$ | Multimodal, separable |
| `Salomon` | $[-100, 100]$ | 0 | $0$ | Multimodal, rotation-invariant |
| `Weierstrass` | $[-0.5, 0.5]$ | 0 | $0$ | Continuous, nowhere differentiable |
| `ExpandedSchafferF6` | $[-100, 100]$ | 0 | $0$ | Multimodal (CEC base function) |
| `HappyCat` | $[-2, 2]$ | 0 | $-1$ | Multimodal (CEC base function) |
| `HGBat` | $[-2, 2]$ | 0 | $-1$ | Multimodal (CEC base function) |

```python
from opytimizer.benchmarks import Rastrigin

problem = Rastrigin(n_variables=10)

problem.optimal_value       # 0.0
problem.optimal_solution()  # array of zeros, on the env's backend
problem(problem.optimal_solution())  # 0.0
```

## CEC 2022 Suite

The twelve functions of the CEC 2022 competition on single-objective bound-constrained optimization, `CEC2022F1` to `CEC2022F12`. They are shifted and rotated versions (or hybrid and composition combinations) of the base functions above, defined on $[-100, 100]^D$:

| Class | Type | Base function(s) | $f^*$ |
| --- | --- | --- | --- |
| `CEC2022F1` | Basic | Zakharov | 300 |
| `CEC2022F2` | Basic | Rosenbrock | 400 |
| `CEC2022F3` | Basic | Schaffer's F7 | 600 |
| `CEC2022F4` | Basic | Non-continuous Rastrigin | 800 |
| `CEC2022F5` | Basic | Levy | 900 |
| `CEC2022F6` | Hybrid | Bent Cigar, HGBat, Rastrigin | 1800 |
| `CEC2022F7` | Hybrid | HGBat, Katsuura, Ackley, Rastrigin, Schwefel, Schaffer's F7 | 2000 |
| `CEC2022F8` | Hybrid | Katsuura, HappyCat, Griewank-Rosenbrock, Schwefel, Ackley | 2200 |
| `CEC2022F9` | Composition | Rosenbrock, Elliptic, Bent Cigar, Discus, Elliptic | 2300 |
| `CEC2022F10` | Composition | Schwefel, Rastrigin, HGBat | 2400 |
| `CEC2022F11` | Composition | Expanded Schaffer's F6, Schwefel, Griewank, Rosenbrock, Rastrigin | 2600 |
| `CEC2022F12` | Composition | HGBat, Rastrigin, Schwefel, Bent Cigar, Elliptic, Expanded Schaffer's F6 | 2700 |

* **Dimensions**: only the dimensions with official data are available: $D \in \{2, 10, 20\}$, and $D \in \{10, 20\}$ for the hybrid functions F6-F8. The default is $D = 10$.
* **Biases**: the official biases are kept, so `optimal_value` is the bias of the function and the error of a run is `fitness - problem.optimal_value`.
* **Data**: the shift vectors, rotation matrices and shuffles ship with the package and are only read when a problem is first evaluated.
* **Reference code**: the implementation follows the organizers' C code, which was used to compute the published results of the competition. Where the code differs from the technical report, the code is followed (the differences are documented in the module docstring).

## Multi- and Many-Objective Families

| Family | Problems | Objectives | Variables (default) | Front shapes |
| --- | --- | --- | --- | --- |
| ZDT | `ZDT1`-`ZDT4`, `ZDT6` | 2 | 30 (10 for ZDT4 and ZDT6) | Convex, concave, disconnected |
| DTLZ | `DTLZ1`-`DTLZ7` | $M \geq 2$ | $M + k - 1$, $k = 5$ (DTLZ1), $10$ (DTLZ2-6), $20$ (DTLZ7) | Linear, concave, degenerate, disconnected |
| WFG | `WFG1`-`WFG9` | $M \geq 2$ | $k + l$, $k = 2(M - 1)$ (4 for $M = 2$), $l = 20$ | Convex, mixed, disconnected, linear, degenerate, concave |
| ZCAT | `ZCAT1`-`ZCAT20` | $M \geq 2$ | 30 | Concave, convex, linear, disconnected, degenerate, hybrid |

ZDT5 is not included, since it is defined over binary strings.

### ZDT

Bi-objective problems in which $f_1$ depends on $x_1$ only and $f_2 = g(x_2, \dots, x_n)\, h(f_1, g)$. ZDT4 has $21^9$ local fronts and ZDT6 a non-uniform density of solutions along its front.

### DTLZ

Scalable problems where the first $M - 1$ variables set the position on the front and the remaining $k$ variables set the distance $g$ to it.

* **DTLZ1** and **DTLZ3** have multimodal distance functions, with many local fronts.
* **DTLZ4** biases the density of solutions ($x_i^{100}$).
* **DTLZ5** and **DTLZ6** have degenerate fronts (curves).
* **DTLZ7** has $2^{M-1}$ disconnected regions.

```python
from opytimizer.benchmarks import DTLZ1

problem = DTLZ1(n_objectives=5)  # n_variables = 5 + 5 - 1 = 9
```

### WFG

The Walking Fish Group toolkit maps $z_i \in [0, 2i]$ to the objectives through chains of shift, bias and reduction transformations. The first `k` variables are position-related and the remaining `l = n_variables - k` are distance-related.

* `k` must be a multiple of `n_objectives - 1`.
* WFG2 and WFG3 require an even `l`.

```python
from opytimizer.benchmarks import WFG9

problem = WFG9(n_variables=24, n_objectives=3, k=4)
```

### ZCAT

Recent scalable problems in which every objective is $f_i = i^2\,(F_i + Z_i)$, where $F_i$ shapes the front and $Z_i$ measures the distance to it. Decision variables are defined on $x_i \in [-i/2, i/2]$, and every problem exposes the features of the original suite:

* **`level`**: difficulty of the distance landscape, from `1` (unimodal) to `6`.
* **`bias`**: introduces a bias towards the Pareto set.
* **`imbalance`**: odd and even objectives use different landscapes.
* **`complicated_pareto_set`**: the Pareto set becomes curved instead of a hyperplane.

```python
from opytimizer.benchmarks import ZCAT12

problem = ZCAT12(n_variables=30, n_objectives=3, level=4, complicated_pareto_set=True)
```

## True Pareto Fronts

`pareto_front(n_points=1000)` returns a sample of the true front, of shape `(P, n_objectives)`, on the problem's backend. It can be used directly as the reference set of metrics such as `IGD`.

* **Linear and spherical fronts** (DTLZ1-4, WFG4-9) are sampled on a Das-Dennis lattice, the largest one with at most `n_points` points.
* **Degenerate fronts** (DTLZ5, DTLZ6, WFG3) are sampled along their analytical curve.
* **Fronts with dominated parts** (ZDT3, DTLZ7, WFG1, WFG2 and ZCAT) are sampled ten times more densely, filtered to their non-dominated points and reduced to `n_points` well-spread points.

:::note
With many objectives ($M > 5$), the position space of fronts with dominated parts is sampled sparsely, so a few of their points may be slightly dominated. Use the samples as an approximation in that regime.
:::

## Writing a Custom Benchmark

A problem of your own does not need to follow this interface: any callable works with `Function`. To follow it anyway, and get the input handling and backend selection for free, subclass the base classes of `opytimizer.benchmarks._base` and implement `_evaluate(X, xp)`, which receives a batch of shape `(N, n_variables)` and the array module (`numpy` or `cupy`).

## References

* M. Jamil and X.-S. Yang, "A literature survey of benchmark functions for global optimization problems," *Int. J. Math. Model. Numer. Optim.*, vol. 4, no. 2, pp. 150-194, 2013.
* A. Kumar, K. V. Price, A. W. Mohamed, A. A. Hadi, and P. N. Suganthan, "Problem definitions and evaluation criteria for the CEC 2022 special session and competition on single objective bound constrained numerical optimization," Technical Report, 2021.
* E. Zitzler, K. Deb, and L. Thiele, "Comparison of multiobjective evolutionary algorithms: Empirical results," *Evolutionary Computation*, vol. 8, no. 2, pp. 173-195, 2000.
* K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, "Scalable test problems for evolutionary multiobjective optimization," in *Evolutionary Multiobjective Optimization*, Springer, 2005, pp. 105-145.
* S. Huband, P. Hingston, L. Barone, and L. While, "A review of multiobjective test problems and a scalable test problem toolkit," *IEEE Trans. Evol. Comput.*, vol. 10, no. 5, pp. 477-506, 2006.
* S. Zapotecas-Martínez, C. A. Coello Coello, H. E. Aguirre, and K. Tanaka, "Challenging test problems for multi- and many-objective optimization," *Swarm and Evolutionary Computation*, vol. 81, art. 101350, 2023.
