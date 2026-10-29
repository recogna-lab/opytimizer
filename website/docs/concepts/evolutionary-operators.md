---
sidebar_position: 10
title: Evolutionary Operators
---

# Evolutionary Operators

Evolutionary operators create new candidate solutions by recombining (crossover) or perturbing (mutation) existing agents. They are passed to evolutionary optimizers, such as `NSGA2`, `MOEAD`, and `RVEA`.

## Crossover Operators

* **`ArithmeticCrossover`**: Generates children as random convex combinations of both parents.
* **`SBXCrossover`**: Simulated Binary Crossover, which mimics the behavior of single-point crossover over real-valued variables. Its distribution index `eta` (default `20`) controls how close children are to their parents.
* **`OnePointCrossover`**: Swaps the tail of both parents after a random cut point.

Every crossover operator shares the following parameters:

* **`rate`**: Probability of applying the crossover to a pair of parents.
* **`gene_rate`**: Probability of recombining each decision variable (continuous operators only).
* **`n_offspring`**: Number of children returned by the operator (`1` or `2`).

### Choosing `gene_rate` for SBX

`SBXCrossover` and `SBXCrossoverTensor` recombine each variable with probability `gene_rate = 0.5` by default, the value of Deb's reference implementation of NSGA-II (also adopted by pymoo and PlatEMO). The remaining variables are copied from the parents, which preserves good combinations of variables already found by the population.

Recombining every variable (`gene_rate = 1.0`) perturbs all coordinates of every child at once and severely slows down convergence, especially on multimodal problems. On DTLZ1 with three objectives, for instance, NSGA-II reaches an IGD about two orders of magnitude worse with `gene_rate = 1.0` than with `0.5` after the same number of generations.

## Mutation Operators

* **`GaussianMutation`**: Adds Gaussian noise with standard deviation `std` to the decision variables.
* **`PolynomialMutation`**: Perturbs the decision variables using a polynomial distribution with distribution index `eta`.
* **`BitFlipMutation`**: Flips binary decision variables, used with boolean spaces.

Every mutation operator has a **`rate`** parameter, which defines the probability of mutating each decision variable. For `PolynomialMutation`, the usual choice is `rate = 1 / n_variables`.

## Tensorized Operators

Each operator has a tensorized counterpart (e.g., `SBXCrossoverTensor`, `PolynomialMutationTensor`), which processes the whole population at once and receives an `Environment` to run on CPU or GPU.

Tensorized optimizers (`NSGA2Tensor`, `MOEADTensor`, `RVEATensor` and `RVEACuda`) bind their operators to the environment of the space when they are compiled, so their default operators work on both NumPy and CuPy spaces.

## Offspring Size

In NSGA-II, each generation creates as many children as there are agents: $\lceil N / 2 \rceil$ pairs of parents are recombined and the union of parents and children ($2N$ solutions) is reduced back to $N$ survivors. Both `NSGA2` and `NSGA2Tensor` follow this scheme, so they perform the same number of evaluations per generation.

## Custom Operators

New operators can be created by inheriting from `BaseCrossover`, `ContinuousCrossover`, or `BaseMutation` and implementing their `__call__` method.
