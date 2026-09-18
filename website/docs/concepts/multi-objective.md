---
sidebar_position: 4
title: "Multi-Objective Concepts"
---

# Multi-Objective Optimization

Multi-objective problems involve evaluating $M$ objectives simultaneously:

* **Objective Vector**: Minimize $F(x) = [f_1(x), f_2(x), \dots, f_M(x)]^T$

## Pareto Dominance

A solution $x_1$ dominates $x_2$ ($x_1 \prec x_2$) if:
1. $x_1$ is no worse than $x_2$ in all objectives.
2. $x_1$ is strictly better than $x_2$ in at least one objective.

The set of non-dominated solutions forms the **Pareto Front**.

## Scalarization and Decomposition

Algorithms like MOEA/D use decomposition functions (such as Penalty-based Boundary Intersection — PBI or Tchebycheff) to reduce a multi-objective problem into $N$ scalar optimization subproblems defined by weight vectors.