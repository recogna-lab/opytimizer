---
title: SPEA2
description: API reference for SPEA2.
---

# `SPEA2`

**Module:** `opytimizer.optimizers.multi_objective.evolutionary.spea2`

References:
    E. Zitzler, M. Laumanns, and L. Thiele. SPEA2: Improving the Strength Pareto
    Evolutionary Algorithm. Technical Report 103, TIK-Report, ETH Zurich (2001).

## Constructor

```python
SPEA2(params: dict = None, archive_size: int = 100, crossover_operator=None, mutation_operator=None) -> None
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `params` | `dict` | `None` | — |
| `archive_size` | `int` | `100` | — |
| `crossover_operator` |  | `None` | — |
| `mutation_operator` |  | `None` | — |

## Methods

### `compile`

```python
compile(self, space: opytimizer.core.space._MultiObjectiveSpace) -> None
```

Compiles additional information required for the optimizer.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | — |

### `update`

```python
update(self, space: opytimizer.core.space._MultiObjectiveSpace) -> None
```

Executes one generation of the SPEA2 algorithm (Algorithm 1, p. 5).

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `opytimizer.core.space._MultiObjectiveSpace` |  | — |
