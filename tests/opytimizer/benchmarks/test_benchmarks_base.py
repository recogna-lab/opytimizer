import numpy as np
import pytest

from opytimizer.benchmarks import DTLZ2, Sphere
from opytimizer.core import Environment, Function
from opytimizer.utils import exception as e


def _gpu_env():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device available")
    return Environment("cupy", "float32")


def test_benchmark_default_env():
    p = Sphere(n_variables=4)

    assert p.env.backend.value == "numpy"
    assert p.dtype == np.float64
    assert p.lower_bound == [-100.0] * 4
    assert p.upper_bound == [100.0] * 4


def test_benchmark_input_shapes():
    p = DTLZ2(n_variables=6, n_objectives=3)
    x = np.full(6, 0.5)

    assert p(x).shape == (3,)
    assert p(x.reshape(6, 1)).shape == (3,)
    assert p(np.tile(x, (5, 1))).shape == (5, 3)
    assert p(np.tile(x, (5, 1))[:, :, None]).shape == (5, 3)


def test_benchmark_single_objective_shapes():
    p = Sphere(n_variables=3)

    assert np.ndim(p(np.ones(3))) == 0
    assert p(np.ones((7, 3))).shape == (7,)


def test_benchmark_wrong_size():
    p = Sphere(n_variables=3)

    with pytest.raises(e.SizeError):
        p(np.ones(4))


def test_benchmark_invalid_arguments():
    with pytest.raises(e.ValueError):
        Sphere(n_variables=0)

    with pytest.raises(e.ValueError):
        Sphere(n_variables=2, lower_bound=1.0, upper_bound=0.0)

    with pytest.raises(e.ValueError):
        DTLZ2().pareto_front(n_points=0)


def test_benchmark_custom_bounds():
    p = Sphere(n_variables=2, lower_bound=[-1.0, -2.0], upper_bound=5.0)

    assert p.lower_bound == [-1.0, -2.0]
    assert p.upper_bound == [5.0, 5.0]


def test_benchmark_with_function():
    p = DTLZ2(n_variables=6, n_objectives=3)
    f = Function(p)

    assert f.name == "DTLZ2"
    assert f(np.full((4, 6, 1), 0.5), xp=np).shape == (4, 3)


def test_benchmark_float32_env():
    p = Sphere(n_variables=3, env=Environment("numpy", "float32"))

    assert p(np.ones((2, 3))).dtype == np.float32


def test_benchmark_gpu():
    env = _gpu_env()
    import cupy as cp

    p = DTLZ2(n_variables=6, n_objectives=3, env=env)
    F = p(cp.full((4, 6), 0.5, dtype=cp.float32))

    assert isinstance(F, cp.ndarray)
    assert isinstance(p.pareto_front(50), cp.ndarray)
    # Host inputs are moved to the device
    assert isinstance(p(np.full((4, 6), 0.5)), cp.ndarray)


def test_benchmark_numpy_env_accepts_gpu_input():
    _gpu_env()
    import cupy as cp

    p = Sphere(n_variables=3)

    assert isinstance(p(cp.ones((2, 3))), np.ndarray)
