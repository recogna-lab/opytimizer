import numpy as np
from opytimizer.spaces import search
from opytimizer.optimizers.multi_objective.evolutionary import moce


def zdt1(x):
    x = x.flatten()
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(max(f1 / g, 0)))
    return np.array([f1, f2])


def test_obmoce_params():
    params = {"DR": 0.5, "CR": 0.9}
    opt = moce.OBMOCE(params=params)
    assert opt.DR == 0.5
    assert opt.CR == 0.9


def test_obmoce_params_setter():
    opt = moce.OBMOCE()
    try:
        opt.DR = "a"
    except:
        opt.DR = 0.5
    try:
        opt.DR = -1
    except:
        opt.DR = 0.5
    assert opt.DR == 0.5

    try:
        opt.CR = "b"
    except:
        opt.CR = 0.9
    try:
        opt.CR = -1
    except:
        opt.CR = 0.9
    assert opt.CR == 0.9


def test_obmoce_logistic():
    opt = moce.OBMOCE()
    x = np.random.random((5, 10))
    out = opt._logistic(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_obmoce_tent():
    opt = moce.OBMOCE()
    x = np.random.random((5, 10))
    out = opt._tent(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_obmoce_gauss():
    opt = moce.OBMOCE()
    x = np.random.random((5, 10))
    out = opt._gauss(x)
    assert out.shape == x.shape
    assert np.all(out > 0)


def test_obmoce_henon():
    opt = moce.OBMOCE()
    opt.y_henon = np.random.random((5, 10))
    x = np.random.random((5, 10))
    x_old = x.copy()
    out = opt._henon(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(opt.y_henon, 0.3 * x_old)


def test_obmoce_opposite():
    opt = moce.OBMOCE()
    lb = np.full(5, 0.0)
    ub = np.full(5, 1.0)
    pos = np.random.random((10, 5))
    op = opt._opposite(pos, lb, ub)
    assert op.shape == pos.shape
    np.testing.assert_allclose(op, lb + ub - pos)
    assert np.all(op >= lb) and np.all(op <= ub)


def test_obmoce_compile():
    opt = moce.OBMOCE()
    space = search.SearchSpace(
        n_agents=10,
        n_variables=5,
        n_objectives=2,
        lower_bound=[0] * 5,
        upper_bound=[1] * 5,
    )
    opt.compile(space)
    assert opt.D.shape == (10, 5)
    assert opt.cp.shape == (10, 5)
    assert opt.y_henon.shape == (10, 5)
    assert set(np.unique(opt.D)) == {-1.0, 1.0}
    assert np.all(opt.cp >= 0) and np.all(opt.cp <= 1)


def test_obmoce_non_dominated_sort():
    opt = moce.OBMOCE()
    fits = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [2.0, 2.0],
    ])
    fronts = opt._non_dominated_sort(fits)
    assert set(fronts[0]) == {0, 1, 2}
    assert set(fronts[1]) == {3}


def test_obmoce_crowding_distance():
    opt = moce.OBMOCE()
    fits = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
    ])
    dist = opt._crowding_distance(fits, [0, 1, 2])
    assert np.isinf(dist[0])
    assert np.isinf(dist[2])
    assert np.isfinite(dist[1])


def test_obmoce_update():
    opt = moce.OBMOCE()
    space = search.SearchSpace(
        n_agents=10,
        n_variables=5,
        n_objectives=2,
        lower_bound=[0] * 5,
        upper_bound=[1] * 5,
    )
    opt.compile(space)
    for ag in space.agents:
        ag.fit = zdt1(ag.position)
    opt.update(space, zdt1)
    assert len(space.agents) == 10
    for ag in space.agents:
        assert np.all(ag.position >= 0)
        assert np.all(ag.position <= 1)