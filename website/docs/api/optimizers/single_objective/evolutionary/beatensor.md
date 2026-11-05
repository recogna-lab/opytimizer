---
title: BEATensor
description: API reference for BEATensor.
---

# `BEATensor`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.bea`

Tensorized (NumPy / CuPy) BlockEA: the same graph, run on the whole population.

## Constructor

```python
BEATensor(params: 'Optional[Dict[str, Any]]' = None, exchange_size: 'int' = 2, crossover_size: 'int' = 3, tournament_sizes: 'Optional[Sequence[int]]' = None, exchange_probabilities: 'Optional[Any]' = None, crossover_scales: 'Optional[Any]' = None, crossover_shifts: 'Optional[Any]' = None, n_offspring: 'Optional[int]' = None, max_generations: 'Optional[int]' = None, de_start: 'float' = 0.7, de_scale: 'float' = 0.5, de_rate: 'float' = 0.9) -> 'None'
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | Contains key-value parameters to the meta-heuristics. |
| `exchange_size` | `int` | `2` | Parents per Exchange offspring (nParents of Algorithm 2). |
| `crossover_size` | `int` | `3` | Parents per Crossover offspring (nParents of Algorithm 3). |
| `tournament_sizes` | `Optional[Sequence[int]]` | `None` | m of each of the 3 Tournament blocks, in [1, 10]. |
| `exchange_probabilities` | `Optional[Any]` | `None` | Probability of each parent in the Exchange blocks, shape (3, exchange_size) (broadcastable). |
| `crossover_scales` | `Optional[Any]` | `None` | w1 of each of the 2 Crossover blocks. |
| `crossover_shifts` | `Optional[Any]` | `None` | w2 of each of the 2 Crossover blocks. |
| `n_offspring` | `Optional[int]` | `None` | Minimum number of offspring per generation (defaults to the population size); it is rounded up so that every division of the graph is exact. |
| `max_generations` | `Optional[int]` | `None` | Number of generations of the run. When given, the late-stage DE is connected after `de_start * max_generations`. |
| `de_start` | `float` | `0.7` | Fraction of the run after which the DE is connected. |
| `de_scale` | `float` | `0.5` | Scaling factor F of the late-stage DE. |
| `de_rate` | `float` | `0.9` | Crossover rate CR of the late-stage DE. |

## Methods

### `compile`

```python
compile(self, space: '_SingleObjectiveTensorSpace') -> 'None'
```

Prepares the block parameters on the space's backend.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveTensorSpace` |  | — |

### `evaluate`

```python
evaluate(self, space: '_SingleObjectiveTensorSpace', function: 'Function') -> 'None'
```

Evaluates the initial population (offspring are evaluated in `update`).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveTensorSpace` |  | — |
| `function` | `Function` |  | — |

### `update`

```python
update(self, space: '_SingleObjectiveTensorSpace', function: 'Function') -> 'None'
```

Performs one generation: offspring creation and environmental selection.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveTensorSpace` |  | A tensorized SingleObjectiveSpace object. |
| `function` | `Function` |  | A Function object that will be used as the objective function. |
