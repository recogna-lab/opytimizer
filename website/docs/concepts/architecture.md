---
sidebar_position: 2
title: Architecture of Opytimizer
---

# Architecture of Opytimizer

Opytimizer is designed around modular, decoupled blocks to ensure extensibility and ease of use.

## Core Modules

* **`Space`**: Manages the population of solutions (agents), their bounds, dimensions, and positions.
* **`Agent`**: Represents an individual candidate solution, holding its coordinate vector and evaluated fitness score.
* **`Optimizer`**: Implements the mathematical logic of the meta-heuristic algorithm responsible for updating agent positions.
* **`Function`**: Encapsulates user-defined objective functions or benchmark functions.
* **`Opytimizer`**: The main orchestration class that controls execution, convergence tracking, and iteration loops.

## Supporting Modules

* **`Environment`**: Selects the array backend (NumPy on CPU or CuPy on GPU) and the numerical precision used by spaces and optimizers.
* **`Stopping Criteria`**: Decide when the optimization loop ends (`MaxIterations`, `MaxEvaluations`, `NoImprovement`).
* **`Callbacks`**: Hook into the optimization pipeline to checkpoint, discretize, or track the task.
* **`History`**: Stores agents, best agents, Pareto fronts, and metric values across iterations.
* **`Hyper-Heuristics`**: High-level optimizers that select, adapt, or generate low-level optimizers.
* **`Benchmarks`**: Test problems from the literature (single-objective, CEC 2022, ZDT, DTLZ, WFG and ZCAT), with their known optima and true Pareto fronts.

## Runner Options

Besides the space, optimizer and function, `Opytimizer` accepts:

* **`save_agents`**: Stores every agent in the history (not only the best one).
* **`save_history`**: Dumps the task's state into the history at every iteration.
* **`enable_logging`**: Whether the library logs to the console and to `opytimizer.log` while the runner is created and during `start()` (default `True`).
* **`show_progress`**: Whether progress bars are drawn during `start()` (default `True`).

## Optimization Pipeline

When `start()` is called, the `Opytimizer` class executes the following loop:

1. Evaluates the initial population.
2. While no stopping criterion is met:
    * Updates the agents' positions using the optimizer.
    * Clips the positions to the search bounds.
    * Evaluates the new population.
    * Dumps the current state into the history.
3. Records the total optimization time.

Depending on the number of objectives defined in the space, `Opytimizer` automatically builds a single-objective or a multi-objective runner.
