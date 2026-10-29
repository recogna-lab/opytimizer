import numpy as np
import pytest

from opytimizer import benchmarks
from opytimizer.benchmarks import cec2022
from opytimizer.core import Environment
from opytimizer.utils import exception as e

CEC2022 = [getattr(benchmarks, f"CEC2022F{i}") for i in range(1, 13)]
CASES = [(cls, d) for cls in CEC2022 for d in cls._dimensions]
REFERENCE_D10 = {
    1: 1172573478.9301, 2: 7048.833808, 3: 705.387214, 4: 957.527,
    5: 5010.684249, 6: 17593892874.43318, 7: 2248.861461, 8: 205207.708776,
    9: 3406.81161, 10: 3538.991234, 11: 9655.648659, 12: 3601.669483,
}  # fmt: skip


def _gpu_env():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device available")
    return Environment("cupy", "float64")


@pytest.mark.parametrize("cls,d", CASES, ids=lambda c: getattr(c, "__name__", c))
def test_cec2022_optimum(cls, d):
    p = cls(d)
    x = p.optimal_solution()

    assert p.lower_bound == [-100.0] * d and p.upper_bound == [100.0] * d
    assert np.all(np.abs(x) <= 100.0)
    assert np.isclose(float(p(x)), p.optimal_value, atol=1e-6)
    assert p.optimal_value == cls._bias


@pytest.mark.parametrize("cls", CEC2022, ids=lambda c: c.__name__)
def test_cec2022_reference_values(cls):
    x = np.linspace(-50.0, 50.0, 10)

    assert np.isclose(float(cls(10)(x)), REFERENCE_D10[cls._function_id], rtol=1e-9)


@pytest.mark.parametrize("cls,d", CASES, ids=lambda c: getattr(c, "__name__", c))
def test_cec2022_batch_matches_single(cls, d):
    p = cls(d)
    X = np.random.default_rng(0).uniform(-100.0, 100.0, (8, d))
    F = p(X)

    assert F.shape == (8,)
    assert np.all(F >= p.optimal_value - 1e-9)
    assert np.allclose(F, [float(p(x)) for x in X])
    assert np.allclose(F, p(X[:, :, None]))


def test_cec2022_dimensions():
    with pytest.raises(e.ValueError):
        benchmarks.CEC2022F1(n_variables=5)

    # Hybrid functions have no data for D = 2
    with pytest.raises(e.ValueError):
        benchmarks.CEC2022F6(n_variables=2)

    assert benchmarks.CEC2022F1().n_variables == 10


def test_cec2022_lazy_data():
    p = benchmarks.CEC2022F12(20)

    assert p._device_data is None
    p(np.zeros(20))
    assert p._device_data is not None


def test_cec2022_composition_local_optima():
    # At the optimum of the i-th component, f = global bias + bias_i
    p = benchmarks.CEC2022F10(10)
    shift, _, _ = cec2022._load_data(10, 10)

    assert np.isclose(float(p(shift[1])), 2400.0 + 200.0)
    assert np.isclose(float(p(shift[2])), 2400.0 + 100.0)


@pytest.mark.parametrize("cls", CEC2022, ids=lambda c: c.__name__)
def test_cec2022_gpu(cls):
    env = _gpu_env()
    import cupy as cp

    d = 10
    X = np.random.default_rng(1).uniform(-100.0, 100.0, (16, d))
    gpu = cls(d, env=env)
    F = gpu(cp.asarray(X))

    assert isinstance(F, cp.ndarray)
    assert np.allclose(F.get(), cls(d)(X), rtol=1e-10)
    assert np.isclose(float(gpu(gpu.optimal_solution())), cls._bias, atol=1e-6)
