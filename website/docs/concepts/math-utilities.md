---
sidebar_position: 16
title: Mathematical Utilities
---

# Mathematical Utilities

The `opytimizer.math` package gathers the mathematical routines shared by optimizers, spaces, and functions.

## Random Numbers

The `opytimizer.math.random` module generates random numbers with a given `size`:

* **`generate_binary_random_number`**: Binary values in $\{0, 1\}$.
* **`generate_uniform_random_number`**: Values drawn from $\mathcal{U}(low, high)$.
* **`generate_gaussian_random_number`**: Values drawn from $\mathcal{N}(\mu, \sigma^2)$.
* **`generate_integer_random_number`**: Integers within a range, optionally excluding a value.
* **`generate_exponential_random_number`**: Values drawn from an exponential distribution.
* **`generate_gamma_random_number`**: Values drawn from a gamma distribution.

## Distributions

The `opytimizer.math.distribution` module generates arrays following specific distributions:

* **`generate_bernoulli_distribution`**: Bernoulli trials with probability `prob`.
* **`generate_choice_distribution`**: Random choices from a vector, optionally weighted by probabilities.
* **`generate_levy_distribution`**: Lévy flights, commonly used by swarm-based optimizers (e.g., Cuckoo Search).

## General Functions

The `opytimizer.math.general` module provides:

* **`euclidean_distance`** and **`angular_distance`**: Distances between two vectors.
* **`kmeans`**: K-Means clustering, used by optimizers that group their agents.
* **`n_wise`**: Iterates over a list in groups of `size` elements.
* **`tournament_selection`**: Selects `n` individuals through tournaments.
* **`weighted_wheel_selection`**: Roulette-wheel selection based on weights.

## Hyper-Complex Functions

The `opytimizer.math.hyper` module maps hyper-complex positions to real values:

* **`norm`**: Norm of each hyper-complex variable.
* **`span`**: Spans the normalized values to the lower and upper bounds.
* **`span_to_hyper_value`**: Decorator that applies `span` to the input of an objective function.

The spanning operation is defined as:

* $\hat{x} = (x_U - x_L) \cdot \frac{\lVert x \rVert}{\sqrt{d}} + x_L$

Where $d$ is the number of dimensions.
