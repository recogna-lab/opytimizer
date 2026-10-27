---
sidebar_position: 15
title: Callbacks and History
---

# Callbacks and History

Callbacks and history allow users to interact with, and inspect, the optimization task without modifying the optimizers.

## Callback Hooks

A `Callback` exposes the following methods, which are called by `Opytimizer` during the optimization pipeline:

* **`on_task_begin(opt_model)`**: Called at the beginning of the task.
* **`on_task_end(opt_model)`**: Called at the end of the task.
* **`on_iteration_begin(iteration, opt_model)`**: Called at the beginning of each iteration.
* **`on_iteration_end(iteration, opt_model)`**: Called at the end of each iteration.
* **`on_evaluate_before(*evaluate_args)`** and **`on_evaluate_after(*evaluate_args)`**: Called around the optimizer's `evaluate` method.
* **`on_update_before(*update_args)`** and **`on_update_after(*update_args)`**: Called around the optimizer's `update` method.

## Built-in Callbacks

* **`CheckpointCallback`**: Saves the whole optimization task to `iter_<iteration>_<file_path>` every `frequency` iterations.
* **`DiscreteSearchCallback`**: Maps continuous positions to the nearest value in a list of `allowed_values`, enabling discrete optimization.
* **`PerformanceTrackingCallback`**: Tracks the performance of low-level optimizers within a sliding window, used with hyper-heuristics.

## History

The `History` class stores the state of the task at each iteration when `save_history=True` is passed to `Opytimizer`:

* **`agents`**: Positions and fitness values of every agent (only when `save_agents=True`).
* **`best_agent`**: Position and fitness value of the best agent (single-objective tasks).
* **`pareto_front`**: Non-dominated solutions (multi-objective tasks).
* **`<metric>`**: Value of each performance metric (multi-objective tasks).
* **`time`**: Total optimization time, in seconds (always stored).

Every key is stored as a list, where the last dimension stands for the iteration. The `get_convergence(key, index)` method reshapes these lists into arrays that are ready to be plotted.

## Saving and Loading

The whole optimization task, including its history, can be saved with `opt.save(file_path)` and restored with `Opytimizer.load(file_path)`. Calling `start()` again continues the task from where it stopped.
