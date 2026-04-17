import numpy as np
from opytimizer.spaces import search
from opytimizer.optimizers.single_objective.evolutionary import ce as obce


def sphere(x):
    return float(np.sum(x ** 2))


def test_obce_params():
    params = {"DR": 0.5, "CR": 0.9}
    opt = obce.OBCE(params=params)
    assert opt.DR == 0.5
    assert opt.CR == 0.9


def test_obce_params_setter():
    opt = obce.OBCE()
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


def test_obce_logistic():
    opt = obce.OBCE()
    x = np.random.random((5, 10))
    out = opt._logistic(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_obce_tent():
    opt = obce.OBCE()
    x = np.random.random((5, 10))
    out = opt._tent(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_obce_gauss():
    opt = obce.OBCE()
    x = np.random.random((5, 10))
    out = opt._gauss(x)
    assert out.shape == x.shape
    assert np.all(out > 0)


def test_obce_henon():
    opt = obce.OBCE()
    opt.y_henon = np.random.random((5, 10))
    x = np.random.random((5, 10))
    x_old = x.copy()
    out = opt._henon(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(opt.y_henon, 0.3 * x_old)


def test_obce_opposite():
    opt = obce.OBCE()
    lb = np.full(5, -100.0)
    ub = np.full(5,  100.0)
    pos = np.random.uniform(-100, 100, (10, 5))
    op = opt._opposite(pos, lb, ub)
    assert op.shape == pos.shape
    np.testing.assert_allclose(op, lb + ub - pos)
    assert np.all(op >= lb) and np.all(op <= ub)


def test_obce_compile():
    opt = obce.OBCE()
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


def test_obce_update():
    opt = obce.OBCE()
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
    fits_before = [ag.fit for ag in space.agents]
    opt.update(space, sphere)
    fits_after = [ag.fit for ag in space.agents]
    for b, a in zip(fits_before, fits_after):
        assert a <= b + 1e-10
    for ag in space.agents:
        assert np.all(ag.position >= -100)
        assert np.all(ag.position <=  100)