---
title: KnEA
description: API reference for KnEA.
---

# `KnEA`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.knea`

Reference:
    Zhang, X., Tian, Y., & Jin, Y. (2014).
    A knee point-driven evolutionary algorithm for many-objective optimization.
    IEEE Transactions on Evolutionary Computation, 19(6), 761-776.

## Constructor

```python
KnEA(params: dict = None, crossover_operator=None, mutation_operator=None, k: int = 3, T: float = 0.5)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `dict` | `None` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |
| `k` | `int` | `3` | — |
| `T` | `float` | `0.5` | — |

## Methods

### `compile`

```python
compile(self, **kwargs)
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `kwargs` |  |  | — |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveSpace, function)
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | — |
| `function` |  |  | — |
