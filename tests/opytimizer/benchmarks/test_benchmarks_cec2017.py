import hashlib
import io
import zipfile

import numpy as np
import pytest

from opytimizer import benchmarks
from opytimizer.benchmarks import _downloads
from opytimizer.core import Environment
from opytimizer.utils import exception as e

IDS = [1] + list(range(3, 31))
CEC2017 = [getattr(benchmarks, f"CEC2017F{i}") for i in IDS]

# f(x) at x = linspace(-50, 50, 10), from a line-by-line port of the official C code
REFERENCE_D10 = {
    1: 15328534674.5, 3: 155818650.37, 4: 3835.82735646, 5: 808.383657273,
    6: 705.387213573, 7: 996.614263292, 8: 968.93268557, 9: 9099.69524853,
    10: 5036.46241422, 11: 174129205.264, 12: 8044419515.36, 13: 233250622.04,
    14: 6155541787.7, 15: 3706488952.7, 16: 4662.49659835, 17: 2968.26305506,
    18: 41915938430.2, 19: 29769682060.0, 20: 2547.7463641, 21: 2933.41979011,
    22: 5292.19180034, 23: 4334.48755217, 24: 3456.35398125, 25: 9578.91592939,
    26: 8662.68528106, 27: 3777.03226362, 28: 5084.3678294, 29: 16770.4582659,
    30: 1947471576.43,
}  # fmt: skip


@pytest.fixture(scope="module")
def cec2017_data():
    """Official data, downloaded on first use; the tests are skipped without it."""
    try:
        return _downloads.cec2017_data_dir()
    except Exception as error:  # pragma: no cover - depends on the network
        pytest.skip(f"CEC 2017 data unavailable: {error!r}")


def _gpu_env():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device available")
    return Environment("cupy", "float64")


@pytest.mark.parametrize("cls", CEC2017, ids=lambda c: c.__name__)
def test_cec2017_optimum(cec2017_data, cls):
    for d in cls._dimensions:
        p = cls(d)
        x = p.optimal_solution()

        assert p.lower_bound == [-100.0] * d and p.upper_bound == [100.0] * d
        assert np.all(np.abs(x) <= 100.0)
        assert np.isclose(float(p(x)), cls._bias, atol=1e-6)
        assert p.optimal_value == cls._bias == 100.0 * cls._function_id


@pytest.mark.parametrize("cls", CEC2017, ids=lambda c: c.__name__)
def test_cec2017_reference_values(cec2017_data, cls):
    x = np.linspace(-50.0, 50.0, 10)

    assert np.isclose(float(cls(10)(x)), REFERENCE_D10[cls._function_id], rtol=1e-9)


@pytest.mark.parametrize("cls", CEC2017, ids=lambda c: c.__name__)
def test_cec2017_batch_matches_single(cec2017_data, cls):
    p = cls(30)
    X = np.random.default_rng(0).uniform(-100.0, 100.0, (6, 30))
    F = p(X)

    assert F.shape == (6,)
    assert np.all(F >= p.optimal_value - 1e-6)
    assert np.allclose(F, [float(p(x)) for x in X])
    assert np.allclose(F, p(X[:, :, None]))


def test_cec2017_dimensions(cec2017_data):
    with pytest.raises(e.ValueError):
        benchmarks.CEC2017F1(n_variables=5)

    # Hybrid functions and compositions of hybrids have no data for D = 2
    with pytest.raises(e.ValueError):
        benchmarks.CEC2017F11(n_variables=2)
    with pytest.raises(e.ValueError):
        benchmarks.CEC2017F29(n_variables=20)

    assert benchmarks.CEC2017F20(n_variables=20).n_variables == 20
    assert benchmarks.CEC2017F1().n_variables == 10
    assert not hasattr(benchmarks, "CEC2017F2")


def test_cec2017_f9_optimum_is_not_the_shift(cec2017_data):
    # The official code uses w = 1 + (z - 1) / 4, so x = o is not optimal
    p = benchmarks.CEC2017F9(10)
    shift = p._host_data()[0][0]

    assert float(p(shift)) > p.optimal_value + 1.0
    assert np.isclose(float(p(p.optimal_solution())), p.optimal_value)


def test_cec2017_composition_local_optima(cec2017_data):
    # At the optimum of the i-th component, f = global bias + bias_i
    p = benchmarks.CEC2017F22(10)
    shift = p._host_data()[0]

    assert np.isclose(float(p(shift[1])), 2200.0 + 100.0, atol=1e-6)


@pytest.mark.parametrize("cls", CEC2017, ids=lambda c: c.__name__)
def test_cec2017_gpu(cec2017_data, cls):
    env = _gpu_env()
    import cupy as cp

    X = np.random.default_rng(1).uniform(-100.0, 100.0, (8, 10))
    gpu = cls(10, env=env)
    F = gpu(cp.asarray(X))

    assert isinstance(F, cp.ndarray)
    assert np.allclose(F.get(), cls(10)(X), rtol=1e-10)


def _zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_downloads_extract_filters_and_flattens(tmp_path):
    payload = _zip({
        "pkg/input_data/shift_data_1.txt": "1 2 3",
        "pkg/input_data/M_1_D2.txt": "1 0 0 1",
        "../../evil/shift_data_2.txt": "4 5 6",
        "pkg/main.cpp": "int main() {}",
    })  # fmt: skip
    target = tmp_path / "cec2017"

    count = _downloads._extract(payload, _downloads._CEC2017_FILES, target)

    assert count == 3
    assert sorted(p.name for p in target.iterdir()) == [
        "M_1_D2.txt",
        "shift_data_1.txt",
        "shift_data_2.txt",
    ]
    # Paths inside the archive are never followed
    assert not (tmp_path.parent / "evil").exists()


def test_downloads_extract_without_data_leaves_nothing(tmp_path):
    target = tmp_path / "cec2017"

    with pytest.raises(e.ValueError):
        _downloads._extract(_zip({"main.cpp": ""}), _downloads._CEC2017_FILES, target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_downloads_checksum(monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        _downloads.urllib.request, "urlopen", lambda *a, **k: Response(b"data")
    )
    digest = hashlib.sha256(b"data").hexdigest()

    assert _downloads._download("https://example.org/x.zip", digest) == b"data"
    with pytest.raises(e.ValueError):
        _downloads._download("https://example.org/x.zip", "0" * 64)


def test_downloads_reuses_existing_data(monkeypatch, tmp_path):
    (tmp_path / "cec2017").mkdir()
    (tmp_path / "cec2017" / "shift_data_1.txt").write_text("0")
    monkeypatch.setattr(_downloads, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        _downloads, "_download", lambda *a: pytest.fail("should not download")
    )

    assert _downloads.cec2017_data_dir() == tmp_path / "cec2017"
