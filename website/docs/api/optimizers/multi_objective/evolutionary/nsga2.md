---
title: NSGA2
description: API reference for NSGA2.
---

# `NSGA2`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.nsga2`

NSGA2 class, inherited from MultiObjectiveOptimizer.

## Constructor

```python
NSGA2(params: 'Dict' = None, crossover_operator=None, mutation_operator=None) -> 'None'
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Dict` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |

## Methods

### `compile`

```python
compile(self, space: '_MultiObjectiveSpace') -> 'None'
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveSpace` |  | — |

### `evaluate`

```python
evaluate(self, space, function)
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` |  |  | A Space object that will be evaluated. |
| `function` |  |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: '_MultiObjectiveSpace', function) -> 'None'
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_MultiObjectiveSpace` |  | — |
| `function` |  |  | — |
