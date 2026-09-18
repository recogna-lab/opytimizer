---
title: SearchSpace
description: API reference for SearchSpace.
---

# `SearchSpace`

**Module:** `opytimizer.spaces.search`

A SearchSpace Factory Class for agents, variables and methods
related to the search space.

## Constructor

```python
SearchSpace(n_agents: int, n_variables: int, n_objectives: int, lower_bound: Union[float, List, Tuple, Any], upper_bound: Union[float, List, Tuple, Any], mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None, tensorized: bool = False) -> Union[opytimizer.spaces.search._SingleObjectiveSearchSpace, opytimizer.spaces.search._MultiObjectiveSearchSpace, opytimizer.spaces.search._SingleObjectiveTensorSearchSpace, opytimizer.spaces.search._MultiObjectiveTensorSearchSpace]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_agents` | `int` |  | — |
| `n_variables` | `int` |  | — |
| `n_objectives` | `int` |  | — |
| `lower_bound` | `Union[float, List, Tuple, Any]` |  | — |
| `upper_bound` | `Union[float, List, Tuple, Any]` |  | — |
| `mapping` | `Optional[List[str]]` | `None` | — |
| `env` | `opytimizer.core.environment.Environment` | `None` | — |
| `tensorized` | `bool` | `False` | — |
