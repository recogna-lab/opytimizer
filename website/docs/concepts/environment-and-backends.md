---
sidebar_position: 9
title: Environment and Backends
---

# Environment and Backends

The `Environment` class defines where and how arrays are computed. It is shared by spaces, agents, optimizers, and operators.

## Backends

* **`numpy`**: CPU execution (default).
* **`cupy`**: GPU execution through CUDA. If CuPy is not installed, the environment falls back to NumPy.

## Properties

* **`backend`**: Current array backend.
* **`dtype`**: Data type of the arrays (e.g., `float32`, `float64`).
* **`xp`**: Proxy to the current array module, allowing backend-agnostic code (`xp.sum`, `xp.random`, etc.).
* **`use_cuda`**: Whether the GPU backend is active.

## Tensorized Execution

Standard spaces store one `Agent` object per solution. Tensorized spaces (`tensorized=True`) store the whole population in a single array, enabling vectorized updates:

* Tensorized spaces must be used with tensorized optimizers (`*Tensor`, `*Cuda`).
* Standard optimizers cannot process tensorized spaces, and vice versa.
* Objective functions used with tensorized spaces receive the whole population, with shape `(n_agents, n_variables)`, and must return one value per agent.

## Example Code

```python
from opytimizer.core import Environment

# CPU-based Environment
cpu_env = Environment().set_backend("numpy").set_dtype("float64")

# CUDA-based Environment
cuda_env = Environment().set_backend("cupy").set_dtype("float32")
```
