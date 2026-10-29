import numpy as np
import pytest

from opytimizer.math.metrics import (
    GD,
    HV,
    IGD,
    R2,
    ErrorRatio,
    MaximumSpread,
    MonteCarloHV,
    Spread,
)


@pytest.fixture
def perfect_front():
    return np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])


@pytest.fixture
def shifted_front():
    return np.array([[1.1, 4.1], [2.1, 3.1], [3.1, 2.1], [4.1, 1.1]])


@pytest.fixture
def empty_front():
    return np.array([])


# --- IGD Metric Tests ---
def test_igd_metric_perfect(perfect_front):
    metric = IGD(pareto_optimal=perfect_front)
    assert metric.name == "IGD"
    assert metric(perfect_front) == 0.0


def test_igd_metric_shifted(perfect_front, shifted_front):
    metric = IGD(pareto_optimal=perfect_front)
    val = metric(shifted_front)
    assert val > 0.0
    assert isinstance(val, float)


def test_igd_metric_empty(perfect_front, empty_front):
    metric = IGD(pareto_optimal=perfect_front)
    assert metric(empty_front) == 0.0

    metric = IGD(pareto_optimal=empty_front)
    assert metric(perfect_front) == 0.0


# --- GD Metric Tests ---
def test_gd_metric_perfect(perfect_front):
    metric = GD(pareto_optimal=perfect_front)
    assert metric.name == "GD"
    assert metric(perfect_front) == 0.0


def test_gd_metric_subset(perfect_front):
    metric = GD(pareto_optimal=perfect_front)
    subset = perfect_front[:2]
    assert metric(subset) == 0.0


def test_gd_metric_bad(perfect_front):
    metric = GD(pareto_optimal=perfect_front)
    bad_front = np.array([[10.0, 10.0]])
    val = metric(bad_front)
    assert val > 0.0


# --- Spread Metric Tests ---
def test_spread_metric_perfect(perfect_front):
    metric = Spread(pareto_optimal=perfect_front)
    assert metric.name == "Spread"
    val = metric(perfect_front)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_spread_metric_bad(perfect_front):
    metric = Spread(pareto_optimal=perfect_front)
    clumped_front = np.array([[2.4, 2.6], [2.6, 2.4]])
    val = metric(clumped_front)
    assert val > 0.0


def test_spread_metric_single_point(perfect_front):
    metric = Spread(pareto_optimal=perfect_front)
    single_point = np.array([[1.0, 4.0]])
    assert metric(single_point) == 0.0


# --- Error Ratio Tests ---
def test_error_ratio_metric_good(perfect_front):
    metric = ErrorRatio(pareto_optimal=perfect_front)
    assert metric.name == "ErrorRatio"
    assert metric(perfect_front) == 0.0


def test_error_ratio_metric_bad(perfect_front):
    metric = ErrorRatio(pareto_optimal=perfect_front)
    bad_front = np.array([[10.0, 10.0], [20.0, 20.0]])
    assert metric(bad_front) == 1.0


def test_error_ratio_metric_mixed(perfect_front):
    metric = ErrorRatio(pareto_optimal=perfect_front)
    mixed_front = np.array([[1.0, 4.0], [10.0, 10.0]])
    assert metric(mixed_front) == 0.5


# --- R2 Metric Tests ---
def test_r2_metric_execution(perfect_front):
    weights = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    metric = R2(weight_vectors=weights)

    assert metric.name == "R2"
    val = metric(perfect_front)
    assert isinstance(val, float)
    assert val >= 0.0


def test_r2_metric_empty():
    weights = np.array([[0.5, 0.5]])
    metric = R2(weight_vectors=weights)
    assert metric(np.array([])) == 0.0


# --- Maximum Spread Tests ---
def test_maximum_spread_metric():
    metric = MaximumSpread()
    assert metric.name == "MaximumSpread"
    front = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert metric(front) == 5.0


def test_maximum_spread_metric_single():
    metric = MaximumSpread()
    assert metric(np.array([[1.0, 1.0]])) == 0.0


# --- Hypervolume Tests ---
def test_hypervolume_metric_execution():
    ref_point = np.array([5.0, 5.0])
    metric = HV(reference_point=ref_point)

    assert metric.name == "HV"
    front = np.array([[2.0, 4.0], [4.0, 2.0]])
    assert metric(front) == 5.0


def test_hypervolume_metric_single_point():
    ref_point = np.array([5.0, 5.0])
    metric = HV(reference_point=ref_point)
    single = np.array([[2.0, 2.0]])
    # A single point dominates the (5 - 2) x (5 - 2) box
    assert metric(single) == 9.0


def test_hypervolume_metric_invalid_dims():
    ref_point = np.array([1, 1, 1])
    metric = HV(reference_point=ref_point)
    front = np.zeros((5, 3))
    assert metric(front) == 1.0


# --- Backend-independent inputs ---
def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device available")
    return cp


class _Agent:
    def __init__(self, fit):
        self.fit = fit


def _metrics(reference):
    return [
        IGD(reference),
        GD(reference),
        Spread(reference),
        ErrorRatio(reference),
        R2(np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])),
        MaximumSpread(),
        HV(reference_point=[5.0, 5.0]),
        MonteCarloHV(n_samples=2000, lower_bound=[0.0, 0.0], upper_bound=[5.0, 5.0]),
    ]


def test_metrics_accept_tensorized_and_agent_fronts(perfect_front, shifted_front):
    positions = np.zeros((len(shifted_front), 3, 1))
    agents = [_Agent(fit) for fit in shifted_front]

    for metric in _metrics(perfect_front):
        expected = metric(shifted_front)

        assert np.isclose(metric((positions, shifted_front)), expected)
        assert np.isclose(metric(agents), expected)


def test_metrics_accept_gpu_fronts(perfect_front, shifted_front):
    cp = _cupy()

    positions = cp.zeros((len(shifted_front), 3, 1))
    front = cp.asarray(shifted_front, dtype=cp.float32)
    agents = [_Agent(fit) for fit in front]

    for metric in _metrics(perfect_front):
        expected = metric(shifted_front)

        assert np.isclose(metric(front), expected, atol=1e-5)
        assert np.isclose(metric((positions, front)), expected, atol=1e-5)
        assert np.isclose(metric(agents), expected, atol=1e-5)
        assert isinstance(metric.pareto_front, np.ndarray)


def test_metrics_accept_gpu_references(perfect_front, shifted_front):
    cp = _cupy()

    reference = cp.asarray(perfect_front)
    assert np.isclose(IGD(reference)(shifted_front), IGD(perfect_front)(shifted_front))
    assert np.isclose(GD(reference)(shifted_front), GD(perfect_front)(shifted_front))
    assert np.isclose(
        HV(reference_point=cp.asarray([5.0, 5.0]))(shifted_front),
        HV(reference_point=[5.0, 5.0])(shifted_front),
    )
    assert np.isclose(
        R2(cp.eye(2), ideal_point=cp.zeros(2), nadir_point=cp.full(2, 5.0))(
            shifted_front
        ),
        R2(np.eye(2), ideal_point=np.zeros(2), nadir_point=np.full(2, 5.0))(
            shifted_front
        ),
    )


# --- Hypervolume correctness for any number of objectives ---
def _hv_inclusion_exclusion(front, ref):
    """Exact hypervolume of a small set by inclusion-exclusion over its subsets."""
    from itertools import combinations

    volume = 0.0
    for k in range(1, len(front) + 1):
        for subset in combinations(front, k):
            corner = np.max(subset, axis=0)
            volume += (-1) ** (k + 1) * np.prod(np.clip(ref - corner, 0.0, None))
    return volume


@pytest.mark.parametrize("n_objectives", [1, 2, 3, 4, 5, 6])
def test_hypervolume_matches_inclusion_exclusion(n_objectives):
    rng = np.random.default_rng(n_objectives)
    ref = np.full(n_objectives, 1.0)

    for _ in range(5):
        front = rng.random((7, n_objectives))
        # Duplicates and points outside the box must not change the volume
        front = np.vstack([front, front[:2], np.full((1, n_objectives), 2.0)])

        expected = _hv_inclusion_exclusion(front[:7], ref)
        assert HV(reference_point=ref)(front) == pytest.approx(expected, rel=1e-10)


def test_hypervolume_ignores_points_outside_the_box():
    metric = HV(reference_point=[1.0, 1.0, 1.0, 1.0])
    front = np.array([[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 1.5]])

    assert metric(front) == pytest.approx(0.5**4)


def test_hypervolume_dimension_mismatch():
    with pytest.raises(ValueError):
        HV(reference_point=[1.0, 1.0])(np.zeros((3, 3)))


# --- Literature definitions ---
def test_spread_uses_extremes_of_the_reference_front():
    t = np.linspace(0, 1, 101)
    reference = np.c_[t, 1 - t]

    # Evenly spread, but covering only the middle 20% of the front
    assert Spread(reference)(reference[40:61]) == pytest.approx(0.8)


def test_maximum_spread_is_the_bounding_box_diagonal():
    front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.9]])

    assert MaximumSpread()(front) == pytest.approx(np.sqrt(1.0**2 + 0.5**2))


def test_monte_carlo_hv_is_unbiased_with_scalar_bounds():
    np.random.seed(0)
    front = np.array([[0.2, 0.6, 0.4], [0.5, 0.3, 0.3], [0.7, 0.2, 0.6]])
    ref = np.ones(3)

    estimate = MonteCarloHV(n_samples=200000, lower_bound=0.0, upper_bound=1.0)(front)

    assert estimate == pytest.approx(_hv_inclusion_exclusion(front, ref), rel=0.02)


def test_distance_metrics_are_exact_for_coincident_fronts():
    front = np.random.default_rng(3).random((200, 3))

    assert IGD(front)(front) == 0.0
    assert GD(front)(front) == 0.0
    assert ErrorRatio(front)(front) == 0.0
