---
sidebar_position: 14
title: Tree and Graph Representations
---

# Tree and Graph Representations

Some problems cannot be represented by a fixed-size numeric vector. Opytimizer supports structured representations, where each agent's position is a `Tree` or a `Graph`.

## Building Blocks

* **`GraphNode`**: A node that holds a value (a primitive, a terminal, or a constant) and its children.
* **`Edge`**: A connection between two nodes.
* **`Graph`**: A set of nodes and edges, which can be directed or undirected.
* **`Tree`**: A rooted graph, where each node is evaluated recursively from its children.
* **`GraphAgent`**: An agent whose position is a `Graph` or a `Tree`.

## Primitive Sets

A `PrimitiveSet` defines the strongly-typed language used to build trees:

* **`Primitive`**: A function node, defined by its input types and its output type (e.g., `add`, `sin`).
* **`Terminal`**: A leaf node with a fixed value (e.g., an input variable `X`).
* **`Ephemeral`**: A leaf node whose value is drawn from a generator when the tree is created (e.g., a random constant).

The `validate()` method checks whether every type can be produced by at least one terminal, ensuring that valid trees can be generated.

## Tree Generation

`TreeSpace` initializes its agents with random trees whose depth lies between `min_depth` and `max_depth`, according to one of the following methods:

* **`full`**: Every branch reaches the maximum depth.
* **`grow`**: Branches may end early with a terminal.
* **`half_and_half`**: Half of the population is created with `full` and half with `grow` (default).

## Graph Generation

`GraphSpace` initializes its agents with random graphs of `n_nodes` nodes, where each edge exists with probability `edge_prob`. The `directed` and `connected` arguments control the graph topology.

## Tree Evaluation

Objective functions used with these spaces receive the agent's `Tree` (or `Graph`) instead of an array. A tree can be evaluated through its `evaluate()` method, and its size can be controlled through its `depth` property.
