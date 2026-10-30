---
title: RVEA
description: API reference for RVEA.
---

# `RVEA`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.rvea`

RVEA class, inherited from MultiObjectiveOptimizer.

The reference vector guided selection keeps one elitist per reference vector
that has at least one associated solution (Algorithm 2), so the population
may have fewer than N agents in some generations, as in the original paper.
Every generation still creates N offspring.

References:
    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization,"
    in IEEE Transactions on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, Oct. 2016, doi: 10.1109/TEVC.2016.2519378.

## Constructor

```python
RVEA(params: Optional[Dict[str, Any]] = None, crossover_operator=None, mutation_operator=None, reference_vectors: numpy.ndarray = None, max_generations: int = 250, alpha: Union[float, int] = 2.0, fr: float = 0.1, **kwargs)
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | Contains key-value parameters to the meta-heuristics. |
| `crossover_operator` |  | `None` | Crossover operator. Defaults to SBX with eta = 30 and crossover probability 1.0, as in Cheng et al. (2016). |
| `mutation_operator` |  | `None` | Mutation operator. Defaults to the polynomial mutation with eta = 20 and rate 1 / n_variables. |
| `reference_vectors` | `numpy.ndarray` | `None` | Reference vectors, one per agent. |
| `max_generations` | `int` | `250` | Number of generations t_max of the run, used by the penalty function and the adaptation frequency. It should match the stopping criterion. |
| `alpha` | `Union[float, int]` | `2.0` | Rate of change of the penalty function. |
| `fr` | `float` | `0.1` | Frequency of the reference vector adaptation (fraction of t_max). |
| `kwargs` |  |  | — |

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
evaluate(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function)
```

Evaluates the search space according to the objective function.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | A Space object that will be evaluated. |
| `function` | `opytimizer.core.function.Function` |  | A Function object serving as an objective function. |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveSpace, function: opytimizer.core.function.Function)
```

Performs one generation: offspring creation, selection and adaptation.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | A MultiObjectiveSpace object. |
| `function` | `opytimizer.core.function.Function` |  | A Function object that will be used as the objective function. |
