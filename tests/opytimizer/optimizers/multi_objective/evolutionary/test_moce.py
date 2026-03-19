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


def test_moce_params():
    params = {"DR": 0.7, "CR": 0.7}
    opt = moce.MOCE(params=params)
    assert opt.DR == 0.7
    assert opt.CR == 0.7


def test_moce_params_setter():
    opt = moce.MOCE()
    try:
        opt.DR = "a"
    except:
        opt.DR = 0.7
    try:
        opt.DR = -1
    except:
        opt.DR = 0.7
    assert opt.DR == 0.7

    try:
        opt.CR = "b"
    except:
        opt.CR = 0.7
    try:
        opt.CR = -1
    except:
        opt.CR = 0.7
    assert opt.CR == 0.7


def test_moce_logistic():
    opt = moce.MOCE()
    x = np.random.random((5, 10))
    out = opt._logistic(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_moce_tent():
    opt = moce.MOCE()
    x = np.random.random((5, 10))
    out = opt._tent(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_moce_gauss():
    opt = moce.MOCE()
    x = np.random.random((5, 10))
    out = opt._gauss(x)
    assert out.shape == x.shape
    assert np.all(out > 0)


def test_moce_henon():
    opt = moce.MOCE()
    opt.y_henon = np.random.random((5, 10))
    x = np.random.random((5, 10))
    x_old = x.copy()
    out = opt._henon(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(opt.y_henon, 0.3 * x_old)



def test_moce_non_dominated_sort():
    opt = moce.MOCE()
    fits = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [2.0, 2.0],
    ])
    fronts = opt._non_dominated_sort(fits)
    assert set(fronts[0]) == {0, 1, 2}
    assert set(fronts[1]) == {3}


def test_moce_crowding_distance():
    opt = moce.MOCE()
    fits = np.array([
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
    ])
    dist = opt._crowding_distance(fits, [0, 1, 2])
    assert np.isinf(dist[0])
    assert np.isinf(dist[2])
    assert np.isfinite(dist[1])


def test_moce_compile():
    opt = moce.MOCE()
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


def test_moce_update():
    opt = moce.MOCE()
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