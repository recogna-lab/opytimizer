import numpy as np
import pytest
from opytimizer.math.metrics import (
    IGD,
    GD,
    Spread,
    ErrorRatio,
    R2,
    MaximumSpread,
    HV,
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
    assert metric(single) == 0.0

def test_hypervolume_metric_invalid_dims():
    ref_point = np.array([1, 1, 1])
    metric = HV(reference_point=ref_point)
    front = np.zeros((5, 3))
    assert metric(front) == 1.0