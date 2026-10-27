---
sidebar_position: 13
title: Hyper-Heuristics
---

# Hyper-Heuristics

A hyper-heuristic is a high-level method that searches the space of heuristics instead of the space of solutions. In Opytimizer, hyper-heuristics manage a set of low-level optimizers and can be used anywhere an `Optimizer` is expected.

## Hyper-Heuristic Types

* **`SelectionHyperHeuristic`**: Periodically selects which low-level optimizer updates the population, according to a selection strategy.
* **`AdaptiveHyperHeuristic`**: Periodically adapts the parameters of the low-level optimizers, according to an adaptation mechanism.
* **`HybridHyperHeuristic`**: Combines both selection and adaptation.
* **`ComponentBasedHyperHeuristic`**: Generates new algorithms by evolving combinations of components (selection, crossover, and mutation functions).

## Selection Strategies

* **`ChoiceFunction`**: Balances recent (`alpha`) and historical (`beta`) performance.
* **`MultiArmedBandit`**: Upper Confidence Bound (UCB) selection, controlled by an `exploration_constant`.
* **`GreedySelection`**: Always selects the best-performing optimizer.
* **`RandomDescent`**: Keeps the current optimizer while it improves, accepting worse performance up to an `acceptance_threshold`.
* **`SimulatedAnnealingSelection`**: Accepts worse optimizers with a probability that decreases over time.

## Adaptation Mechanisms

* **`ParameterAdaptation`**: Adjusts optimizer parameters by an `adaptation_rate`.
* **`StrategyAdaptation`**: Changes strategies when the improvement falls below an `adaptation_threshold`.
* **`PopulationAdaptation`**: Adjusts the population size between `min_population` and `max_population`.
* **`AdaptiveParameterControl`**: Adapts parameters based on a window of recent performance.
* **`SelfAdaptiveControl`**: Learns parameter adjustments with a `learning_rate`.

## Performance Metric

Every hyper-heuristic accepts a `performance_metric` callable, which receives the list of agents and returns a value used to rank the low-level optimizers (e.g., `min(agent.fit for agent in agents)`). If no metric is provided, the performance history is not updated.

## Intervals

* **`selection_interval`**: Number of iterations between optimizer selections.
* **`adaptation_interval`**: Number of iterations between parameter adaptations.
