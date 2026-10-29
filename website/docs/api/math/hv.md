---
title: HV
description: API reference for HV.
---

# `HV`

**Module:** `opytimizer.math.metrics`

Exact hypervolume, for any number of objectives (minimization).

The hypervolume is the volume of the objective space dominated by the front
and bounded by `reference_point`. Points that are not strictly better than
the reference point in every objective do not contribute to it.

## Constructor

```python
HV(reference_point=[1.01, 1.01])
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `reference_point` |  | `[1.01, 1.01]` | — |
