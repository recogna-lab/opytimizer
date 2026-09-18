---
title: TSPACO
description: API reference for TSPACO.
---

# `TSPACO`

**Module:** `opytimizer.optimizers.single_objective.swarm.aco`

Ant Colony Optimization specialized for the Traveling Salesman Problem.

Each ant constructs a Hamiltonian cycle by selecting the next
unvisited city according to pheromone and heuristic information.

## Constructor

```python
TSPACO(params: Optional[Dict[str, Any]] = None, distance_matrix: Optional[numpy.ndarray] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | Contains key-value parameters to the meta-heuristic. |
| `distance_matrix` | `Optional[numpy.ndarray]` | `None` | Matrix containing pairwise distances between cities. |

## Methods

### `compile`

```python
compile(self, space: opytimizer.spaces.graph._SingleObjectiveGraphSpace) -> None
```

Compiles pheromone and TSP heuristic information.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.spaces.graph._SingleObjectiveGraphSpace` |  | — |
