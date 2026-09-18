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