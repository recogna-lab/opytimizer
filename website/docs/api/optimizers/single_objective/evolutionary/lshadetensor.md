---
title: LSHADETensor
description: API reference for LSHADETensor.
---

# `LSHADETensor`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.lshade`

Agnostic Tensorized L-SHADE Implementation (NumPy/CuPy).

## Constructor

```python
LSHADETensor(params: 'Dict' = None, MAX_NFE: 'int' = 100, H: 'int' = 100, p: 'float' = 0.11, f_arc: 'float' = 2.6)
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
compile(self, space: '_SingleObjectiveTensorSpace', **kwargs)
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveTensorSpace` |  | — |
| `kwargs` |  |  | — |

### `evaluate`

```python
evaluate(self, space: '_SingleObjectiveTensorSpace', function: 'Function')
```

Evaluates the search space according to the objective function.

If you need a specific evaluate method, please re-implement
it on child's class.

Also, note that function only accept arguments that are
found on Opytimizer class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveTensorSpace` |  | A Space object that will be evaluated. |
| `function` | `Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: '_SingleObjectiveTensorSpace', function: 'Function')
```

Updates the agents' position array.

As each child has a different procedure of update, you will need
to implement it directly on its class.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveTensorSpace` |  | — |
| `function` | `Function` |  | — |
