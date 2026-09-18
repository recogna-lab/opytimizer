---
title: LSHADE
description: API reference for LSHADE.
---

# `LSHADE`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.lshade`

References:
    R. Tanabe and A. S. Fukunaga, "Improving the search performance of SHADE using linear population size reduction,"
    2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, China, 2014, pp. 1658-1665, doi: 10.1109/CEC.2014.6900380.

## Constructor

```python
LSHADE(params: 'Dict' = None, MAX_NFE: 'int' = 100, H: 'int' = 100, p: 'float' = 0.11, f_arc: 'float' = 2.6)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Dict` | `None` | — |
| `MAX_NFE` | `int` | `100` | — |
| `H` | `int` | `100` | — |
| `p` | `float` | `0.11` | — |
| `f_arc` | `float` | `2.6` | — |

## Methods

### `compile`

```python
compile(self, space: '_SingleObjectiveSpace', **kwargs)
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveSpace` |  | — |
| `kwargs` |  |  | — |

### `evaluate`

```python
evaluate(self, space, function)
```

Evaluates the search space according to the objective function.

If you need a specific evaluate method, please re-implement
it on child's class.

Also, note that function only accept arguments that are
found on Opytimizer class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` |  |  | A Space object that will be evaluated. |
| `function` |  |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: '_SingleObjectiveSpace', function: 'Function')
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveSpace` |  | — |
| `function` | `Function` |  | — |
