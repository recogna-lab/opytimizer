---
sidebar_position: 10
title: Evolutionary Operators
---

# Evolutionary Operators

Evolutionary operators create new candidate solutions by recombining (crossover) or perturbing (mutation) existing agents. They are passed to evolutionary optimizers, such as `NSGA2`, `MOEAD`, and `RVEA`.

## Crossover Operators

* **`ArithmeticCrossover`**: Generates children as random convex combinations of both parents.
* **`SBXCrossover`**: Simulated Binary Crossover, which mimics the behavior of single-point crossover over real-valued variables. Its distribution index `eta` controls how close children are to their parents.
* **`OnePointCrossover`**: Swaps the tail of both parents after a random cut point.

Every crossover operator shares the following parameters:

* **`rate`**: Probability of applying the crossover to a pair of parents.
* **`gene_rate`**: Probability of recombining each decision variable (continuous operators only).
* **`n_offspring`**: Number of children returned by the operator (`1` or `2`).

## Mutation Operators

* **`GaussianMutation`**: Adds Gaussian noise with standard deviation `std` to the decision variables.
* **`PolynomialMutation`**: Perturbs the decision variables using a polynomial distribution with distribution index `eta`.
* **`BitFlipMutation`**: Flips binary decision variables, used with boolean spaces.

Every mutation operator has a **`rate`** parameter, which defines the probability of mutating each decision variable.

## Tensorized Operators

Each operator has a tensorized counterpart (e.g., `SBXCrossoverTensor`, `PolynomialMutationTensor`), which processes the whole population at once and receives an `Environment` to run on CPU or GPU.

## Custom Operators

New operators can be created by inheriting from `BaseCrossover`, `ContinuousCrossover`, or `BaseMutation` and implementing their `__call__` method.
