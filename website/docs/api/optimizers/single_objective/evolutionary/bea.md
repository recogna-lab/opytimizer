---
title: BEA
description: API reference for BEA.
---

# `BEA`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.bea`

Block Evolutionary Algorithm (BlockEA), single-objective.

Three Tournament blocks select parents, which are divided among three
Exchange blocks (each variable copied from one of `exchange_size` parents);
half of the output of Exchange 1 and 2 feeds Crossover 1 and the output of
Exchange 3 feeds Crossover 2, both combining `crossover_size` parents with
weights that sum to one. The environmental selection keeps the N best of
the population and the offspring. In the late stage, a DE/current-to-best/
1/bin is connected to speed up the convergence.

The report does not list the trained block parameters nor the details of the
late-stage DE, so they are exposed as arguments.

References:
    X. Qi, Z. Yang, S. Zhu, H. Geng, S. Yang, and Y. Tian, "Report for Block
    Evolutionary Algorithm," competition report.

## Constructor

```python
BEA(params: 'Optional[Dict[str, Any]]' = None, exchange_size: 'int' = 2, crossover_size: 'int' = 3, tournament_sizes: 'Optional[Sequence[int]]' = None, exchange_probabilities: 'Optional[Any]' = None, crossover_scales: 'Optional[Any]' = None, crossover_shifts: 'Optional[Any]' = None, n_offspring: 'Optional[int]' = None, max_generations: 'Optional[int]' = None, de_start: 'float' = 0.7, de_scale: 'float' = 0.5, de_rate: 'float' = 0.9) -> 'None'
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
compile(self, space: '_SingleObjectiveSpace') -> 'None'
```

Prepares the block parameters for the space.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveSpace` |  | — |

### `evaluate`

```python
evaluate(self, space: '_SingleObjectiveSpace', function: 'Function') -> 'None'
```

Evaluates the initial population (offspring are evaluated in `update`).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveSpace` |  | — |
| `function` | `Function` |  | — |

### `update`

```python
update(self, space: '_SingleObjectiveSpace', function: 'Function') -> 'None'
```

Performs one generation: offspring creation and environmental selection.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveSpace` |  | A SingleObjectiveSpace object. |
| `function` | `Function` |  | A Function object that will be used as the objective function. |
