---
title: RVEATensor
description: API reference for RVEATensor.
---

# `RVEATensor`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.rvea`

Backend-agnostic (NumPy/CuPy) tensorized implementation of RVEA.

It follows the same algorithm as `RVEA`: only the K survivors of the last
selection (one per non-empty subpopulation) take part in mating, translation
and adaptation. Since tensors have a fixed shape, the population arrays keep
N rows: the first K hold the survivors and the remaining ones repeat them,
without any device -> host synchronization.

## Constructor

```python
RVEATensor(params: Optional[Dict[str, Any]] = None, crossover_operator=None, mutation_operator=None, reference_vectors: numpy.ndarray = None, max_generations: int = 250, alpha: Union[float, int] = 2.0, fr: float = 0.1)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | Contains key-value parameters to the meta-heuristics. |
| `crossover_operator` |  | `None` | Tensorized crossover operator. Defaults to SBX with eta = 30 and crossover probability 1.0, as in Cheng et al. (2016). |
| `mutation_operator` |  | `None` | Tensorized mutation operator. Defaults to the polynomial mutation with eta = 20 and rate 1 / n_variables. |
| `reference_vectors` | `numpy.ndarray` | `None` | Reference vectors, one per agent. |
| `max_generations` | `int` | `250` | Number of generations t_max of the run, used by the penalty function and the adaptation frequency. It should match the stopping criterion. |
| `alpha` | `Union[float, int]` | `2.0` | Rate of change of the penalty function. |
| `fr` | `float` | `0.1` | Frequency of the reference vector adaptation (fraction of t_max). |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._MultiObjectiveSpace)
```

Compiles additional information that is used by this optimizer.

This method is called before the optimization procedure and makes sure
that the additional variable is available as a property.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | — |

### `evaluate`

```python
evaluate(self, space: opytimizer.core.space._MultiObjectiveTensorSpace, function: opytimizer.core.function.Function) -> None
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveTensorSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveTensorSpace, function: opytimizer.core.function.Function) -> None
```

Performs one generation: offspring creation, selection and adaptation.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveTensorSpace` |  | A tensorized MultiObjectiveSpace object. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
