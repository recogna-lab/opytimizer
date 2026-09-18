---
title: GridSpace
description: API reference for GridSpace.
---

# `GridSpace`

**Module:** `opytimizer.spaces.grid`

A GridSpace Factory Class for agents, variables and methods
related to the grid search space.

## Constructor

```python
GridSpace(n_variables: int, n_objectives: int, step: Union[float, List, Tuple, Any], lower_bound: Union[float, List, Tuple, Any], upper_bound: Union[float, List, Tuple, Any], mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None, tensorized: bool = False) -> Union[opytimizer.spaces.grid._SingleObjectiveGridSpace, opytimizer.spaces.grid._MultiObjectiveGridSpace]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_variables` | `int` |  | — |
| `n_objectives` | `int` |  | — |
| `step` | `Union[float, List, Tuple, Any]` |  | — |
| `lower_bound` | `Union[float, List, Tuple, Any]` |  | — |
| `upper_bound` | `Union[float, List, Tuple, Any]` |  | — |
| `mapping` | `Optional[List[str]]` | `None` | — |
| `env` | `opytimizer.core.environment.Environment` | `None` | — |
| `tensorized` | `bool` | `False` | — |
