---
title: AGGEA
description: API reference for AGGEA.
---

# `AGGEA`

**Module:** `opytimizer.optimizers.single_objective.evolutionary.aggea`

Automatic Generated Graph Evolutionary Algorithm (single-objective).

The population flows through a graph of blocks: Tournament blocks select
parents, Exchange blocks copy each variable from one of n_e parents, Crossover
blocks combine n_c parents with Gaussian-mixture weights and Mutation blocks
perturb the offspring; a (mu + lambda) Selection keeps the N best solutions.
Every variation block is a translation, scale and rotation invariant operator.

The report does not list the trained block parameters, so they are exposed
as arguments; the defaults reproduce well-known operators (binary
tournaments, uniform crossover, DE/rand/1/bin and Gaussian mutation).

References:
    S. Zhu, H. Geng, X. Liu, Z. Zhu, R. Zhang, H. Jiang, C. He, and Y. Tian,
    "Report for the Automatic Generated Graph Evolutionary Algorithm,"
    GNBG competition report, 2025.

## Constructor

```python
AGGEA(params: 'Optional[Dict[str, Any]]' = None, n_tournaments: 'int' = 10, n_exchanges: 'int' = 10, n_crossovers: 'int' = 2, n_mutations: 'int' = 2, exchange_size: 'int' = 2, crossover_size: 'int' = 3, tournament_sizes: 'Optional[Sequence[int]]' = None, exchange_probabilities: 'Optional[Any]' = None, crossover_means: 'Optional[Any]' = None, crossover_stds: 'Optional[Any]' = None, crossover_weights: 'Optional[Any]' = None, mutation_stds: 'Optional[Any]' = None, mutation_weights: 'Optional[Any]' = None, crossover_split: 'Optional[Any]' = None, n_offspring: 'Optional[int]' = None) -> 'None'
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `Optional[Dict[str, Any]]` | `None` | Contains key-value parameters to the meta-heuristics. |
| `n_tournaments` | `int` | `10` | Number of Tournament blocks. |
| `n_exchanges` | `int` | `10` | Number of Exchange blocks. |
| `n_crossovers` | `int` | `2` | Number of Crossover blocks. |
| `n_mutations` | `int` | `2` | Number of Mutation blocks. |
| `exchange_size` | `int` | `2` | Parents per Exchange offspring (n_e). |
| `crossover_size` | `int` | `3` | Parents per Crossover offspring (n_c). |
| `tournament_sizes` | `Optional[Sequence[int]]` | `None` | m of each Tournament block, in [1, 10]. |
| `exchange_probabilities` | `Optional[Any]` | `None` | (p_1, ..., p_ne) of each Exchange block. |
| `crossover_means` | `Optional[Any]` | `None` | Means of the weights (w_2, ..., w_nc), of shape (n_crossovers, components, n_c - 1) (broadcastable). |
| `crossover_stds` | `Optional[Any]` | `None` | Standard deviations with the same shape. |
| `crossover_weights` | `Optional[Any]` | `None` | Probability of each mixture component per block. |
| `mutation_stds` | `Optional[Any]` | `None` | Standard deviations of w_0 per block and component. |
| `mutation_weights` | `Optional[Any]` | `None` | Probability of each mutation component per block. Defaults to a perturbation probability of 1 / n_variables. |
| `crossover_split` | `Optional[Any]` | `None` | Ratio of each Crossover block's offspring sent to each Mutation block. |
| `n_offspring` | `Optional[int]` | `None` | Minimum number of offspring per generation (defaults to the population size); it is rounded up so that every division of the graph is exact. |

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

Performs one generation: offspring creation and (mu + lambda) selection.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `_SingleObjectiveSpace` |  | A SingleObjectiveSpace object. |
| `function` | `Function` |  | A Function object that will be used as the objective function. |
