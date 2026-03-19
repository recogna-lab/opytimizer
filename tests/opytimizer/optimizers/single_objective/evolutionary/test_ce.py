import numpy as np
from opytimizer.spaces import search
from opytimizer.optimizers.single_objective.evolutionary import ce


def sphere(x):
    return float(np.sum(x ** 2))


def test_ce_params():
    params = {"DR": 0.7, "CR": 0.7, "jump": 10}
    opt = ce.CE(params=params)
    assert opt.DR == 0.7
    assert opt.CR == 0.7
    assert opt.jump == 10


def test_ce_params_setter():
    opt = ce.CE()
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

    try:
        opt.jump = "c"
    except:
        opt.jump = 10
    try:
        opt.jump = -1
    except:
        opt.jump = 10
    assert opt.jump == 10


def test_ce_logistic():
    opt = ce.CE()
    x = np.random.random((5, 10))
    out = opt._logistic(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_ce_tent():
    opt = ce.CE()
    x = np.random.random((5, 10))
    out = opt._tent(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_ce_gauss():
    opt = ce.CE()
    x = np.random.random((5, 10))
    out = opt._gauss(x)
    assert out.shape == x.shape
    assert np.all(out > 0)


def test_ce_henon():
    opt = ce.CE()
    opt.y_henon = np.random.random((5, 10))
    x = np.random.random((5,10))
    x_old = x.copy()
    for i in range(5):
        out = opt._henon(x=x[i], i=i)
        assert out.shape == x[i].shape
        np.testing.assert_allclose(opt.y_henon[i], 0.3 * x_old[i])


def test_ce_compile():
    opt = ce.CE()
    space = search.SearchSpace(
        n_agents=10,
        n_variables=5,
        n_objectives=1,
        lower_bound=[-100] * 5,
        upper_bound=[ 100] * 5,
    )
    opt.compile(space)
    assert opt.D.shape == (10, 5)
    assert opt.cp.shape == (10, 5)
    assert opt.y_henon.shape == (10, 5)
    assert set(np.unique(opt.D)) == {-1.0, 1.0}
    assert np.all(opt.cp >= 0) and np.all(opt.cp <= 1)


def test_ce_update():
    opt = ce.CE()
    space = search.SearchSpace(
        n_agents=10,
        n_variables=5,
        n_objectives=1,
        lower_bound=[-100] * 5,
        upper_bound=[ 100] * 5,
    )
    opt.compile(space)
    for ag in space.agents:
        ag.fit = sphere(ag.position)
    opt.update(space, sphere)
    assert len(space.agents) == 10
    for ag in space.agents:
        assert np.all(ag.position >= -100)
        assert np.all(ag.position <=  100)