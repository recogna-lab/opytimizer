---
title: HyperComplexSpace
description: API reference for HyperComplexSpace.
---

# `HyperComplexSpace`

**Module:** `opytimizer.spaces.hyper_complex`

An HyperComplexSpace Factory Class that will hold agents, variables and methods
related to the hypercomplex search space.

## Constructor

```python
HyperComplexSpace(n_agents: int, n_variables: int, n_dimensions: int, n_objectives: int, mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None, tensorized: bool = False) -> Union[opytimizer.spaces.hyper_complex._SingleObjectiveHyperComplexSpace, opytimizer.spaces.hyper_complex._MultiObjectiveHyperComplexSpace, opytimizer.spaces.hyper_complex._SingleObjectiveTensorHyperComplexSpace, opytimizer.spaces.hyper_complex._MultiObjectiveTensorHyperComplexSpace]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_agents` | `int` |  | — |
| `n_variables` | `int` |  | — |
| `n_dimensions` | `int` |  | — |
| `n_objectives` | `int` |  | — |
| `mapping` | `Optional[List[str]]` | `None` | — |
| `env` | `opytimizer.core.environment.Environment` | `None` | — |
| `tensorized` | `bool` | `False` | — |
