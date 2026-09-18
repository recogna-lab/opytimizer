---
title: ACO
description: API reference for ACO.
---

# `ACO`

**Module:** `opytimizer.optimizers.single_objective.swarm.aco`

An ACO class, inherited from Optimizer.

References:
   M. Dorigo, V. Maniezzo and A. Colorni, "Ant system: optimization by a colony of cooperating agents,"
    in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics),
    vol. 26, no. 1, pp. 29-41, Feb. 1996, doi: 10.1109/3477.484436.

## Constructor

```python
ACO(params: Optional[Dict[str, Any]] = None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.spaces.graph._SingleObjectiveGraphSpace) -> None
```

Compiles additional information that is used by this optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.spaces.graph._SingleObjectiveGraphSpace` |  | A non-tensorized, single-objective GraphSpace object. |

### `evaluate`

```python
evaluate(self, space: opytimizer.spaces.graph._SingleObjectiveGraphSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.spaces.graph._SingleObjectiveGraphSpace` |  | A GraphSpace object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |

### `update`

```python
update(self, space: opytimizer.spaces.graph._SingleObjectiveGraphSpace) -> None
```

Wraps Ant System over the whole colony: evaporates/deposits
pheromone based on the --current-- generation's graphs, then has
every ant construct a new graph for the next one.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.spaces.graph._SingleObjectiveGraphSpace` |  | GraphSpace containing agents and update-related information. |
