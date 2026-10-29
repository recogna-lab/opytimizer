---
sidebar_position: 5
title: GPU Acceleration
---

# GPU-Accelerated Optimization

Accelerate single-objective and multi-objective optimization tasks using CuPy and GPU tensor backends.

## Single-Objective GPU Optimization

```python
from opytimizer import Opytimizer
from opytimizer.benchmarks import Sphere
from opytimizer.core import Environment, Function, NoImprovement
from opytimizer.optimizers.single_objective.swarm import PSOCuda
from opytimizer.spaces import SearchSpace

n_agents = 60
n_variables = 30
n_objectives = 1

lower_bound = [-30.0] * n_variables
upper_bound = [30.0] * n_variables

device_gpu = Environment().set_backend("cupy").set_dtype("float32")

# Sphere benchmark function, evaluated on the GPU over the whole population
sphere = Sphere(n_variables, lower_bound, upper_bound, env=device_gpu)

my_search_space = SearchSpace(
    n_agents,
    n_variables,
    n_objectives,
    lower_bound,
    upper_bound,
    env=device_gpu,
    tensorized=True,
)

stopping_criteria = NoImprovement(patience=10, min_delta=1e-5)
my_func = Function(sphere)

my_pso = PSOCuda()

opt = Opytimizer(space=my_search_space, optimizer=my_pso, function=my_func)

opt.start(stopping_criteria=stopping_criteria)
```

Any callable that receives the whole population (shape `(n_agents, n_variables)`) and returns one value per agent works as well, e.g., `lambda x: cp.sum(cp.square(x), axis=1)`.

## Reducing Host-Side Overhead

Most of the time of a GPU run is spent on the device, but a few features of the optimization loop run on the host:

* **`show_progress=False`**: Disables the progress bars, so the displayed values (such as the best fitness) are never copied back from the GPU.
* **`enable_logging=False`**: Disables the console and file logs of the library during the run.
* **`metrics_every`**: In multi-objective runs, recomputes the Pareto front and the metrics only every few iterations, e.g., `opt.start(MaxIterations(250), metrics=[...], metrics_every=10)`. Recomputing the front at every iteration (`save_history=True` or metrics) is usually the most expensive of these features.
* **`NoImprovement(..., check_every=10)`**: Reads the non-improvement counter back from the GPU only every few iterations.

```python
opt = Opytimizer(
    space=my_search_space,
    optimizer=my_pso,
    function=my_func,
    enable_logging=False,
    show_progress=False,
)
```

Progress bars and logs are already throttled, so disabling them changes the run time only slightly (within a few percent in our measurements); they are mostly useful to keep the output clean in long benchmark studies.
