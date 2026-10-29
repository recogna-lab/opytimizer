---
title: Opytimizer
description: API reference for Opytimizer.
---

# `Opytimizer`

**Module:** `opytimizer.opytimizer`

Factory that builds the runner matching the number of objectives.

## Constructor

```python
Opytimizer(space: Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace], optimizer: opytimizer.core.optimizer.Optimizer, function: opytimizer.core.function.Function, save_agents: bool = False, save_history: bool = False, enable_logging: bool = True, show_progress: bool = True) -> opytimizer.opytimizer._BaseRunner
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `space` | `Union[opytimizer.core.space._SingleObjectiveSpace, opytimizer.core.space._MultiObjectiveSpace]` |  | — |
| `optimizer` | `opytimizer.core.optimizer.Optimizer` |  | — |
| `function` | `opytimizer.core.function.Function` |  | — |
| `save_agents` | `bool` | `False` | — |
| `save_history` | `bool` | `False` | — |
| `enable_logging` | `bool` | `True` | — |
| `show_progress` | `bool` | `True` | — |
