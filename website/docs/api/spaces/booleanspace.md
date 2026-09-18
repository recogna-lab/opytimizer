---
title: BooleanSpace
description: API reference for BooleanSpace.
---

# `BooleanSpace`

**Module:** `opytimizer.spaces.boolean`

A BooleanSpace Factory Class for agents, variables and methods
related to the boolean search space.

## Constructor

```python
BooleanSpace(n_agents: int, n_variables: int, n_objectives: int, mapping: Optional[List[str]] = None, env: opytimizer.core.environment.Environment = None, tensorized: bool = False) -> Union[opytimizer.spaces.boolean._SingleObjectiveBooleanSpace, opytimizer.spaces.boolean._MultiObjectiveBooleanSpace, opytimizer.spaces.boolean._SingleObjectiveTensorBooleanSpace, opytimizer.spaces.boolean._MultiObjectiveTensorBooleanSpace]
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_agents` | `int` |  | — |
| `n_variables` | `int` |  | — |
| `n_objectives` | `int` |  | — |
| `mapping` | `Optional[List[str]]` | `None` | — |
| `env` | `opytimizer.core.environment.Environment` | `None` | — |
| `tensorized` | `bool` | `False` | — |
