import numpy as np
import pytest

from opytimizer import Opytimizer
from opytimizer.benchmarks import Rastrigin, Sphere
from opytimizer.core import Environment, Function
from opytimizer.core.stopping import MaxEvaluations, MaxIterations
from opytimizer.optimizers.single_objective.evolutionary import bea
from opytimizer.spaces.search import SearchSpace
from opytimizer.utils import exception as e


def test_bea_defaults():
    opt = bea.BEA()

    assert (opt.exchange_size, opt.crossover_size) == (2, 3)
    assert np.all(opt.tournament_sizes == 2)
    assert np.allclose(opt.exchange_probabilities, 0.5)
    assert opt.max_generations is None
    assert not opt._de_active()


def test_bea_invalid_parameters():
    with pytest.raises(e.ValueError):
        bea.BEA(tournament_sizes=[2, 2])

    with pytest.raises(e.ValueError):
        bea.BEA(crossover_size=1)

    with pytest.raises(e.ValueError):
        bea.BEA(de_start=1.5)

    with pytest.raises(e.ValueError):
        bea.BEA(exchange_probabilities=[-1.0, 2.0])


@pytest.mark.parametrize("n_agents", [10, 50, 100, 257])
@pytest.mark.parametrize("sizes", [(2, 3), (3, 2), (1, 4)])
def test_bea_graph_sizes_divide_exactly(n_agents, sizes):
    opt = bea.BEA(exchange_size=sizes[0], crossover_size=sizes[1])
    t, n_offspring = opt._graph_sizes(n_agents)
    n_e, n_c = sizes

    assert n_offspring >= n_agents
    assert t % 3 == 0 and t % n_e == 0
    assert (t // n_e) % (2 * n_c) == 0
    assert n_offspring == 2 * t // (n_e * n_c)


def test_bea_offspring_shape_and_bounds():
    np.random.seed(0)
    opt = bea.BEA()
    cfg = opt._prepare(60, np, float)
    X = np.random.uniform(-2, 2, (60, 4))

    O = bea._bea_offspring(X, np.sum(X**2, axis=1), np.full(4, -2.0), np.full(4, 2.0), cfg, np)

    assert O.shape == (opt.n_offspring_per_generation, 4)
    assert np.all((O >= -2.0) & (O <= 2.0))


def test_bea_crossover_weights_sum_to_one():
    np.random.seed(0)
    # A population of identical solutions is a fixed point of every block, since
    # the crossover weights of each offspring sum to one
    opt = bea.BEA(crossover_scales=2.0, crossover_shifts=0.3)
    cfg = opt._prepare(30, np, float)
    X = np.tile([[0.25, -0.5, 0.75]], (30, 1))

    O = bea._bea_offspring(X, np.zeros(30), np.full(3, -1.0), np.full(3, 1.0), cfg, np)

    assert np.allclose(O, X[0])


def test_bea_late_stage_de():
    opt = bea.BEA(max_generations=10, de_start=0.5)

    opt._generation = 4
    assert not opt._de_active()
    opt._generation = 5
    assert opt._de_active()

    np.random.seed(0)
    X = np.random.uniform(-1, 1, (20, 3))
    V = bea._de_offspring(X, np.sum(X**2, axis=1), -np.ones(3), np.ones(3), 0.5, 0.9, np)
    assert V.shape == X.shape and np.all(np.abs(V) <= 1.0)


@pytest.mark.parametrize("tensorized", [False, True])
def test_bea_is_elitist_and_converges(tensorized):
    np.random.seed(0)
    env = Environment("numpy", "float64") if tensorized else None
    problem = Sphere(n_variables=5, env=env)
    kw = {"env": env, "tensorized": True} if tensorized else {}
    space = SearchSpace(50, 5, 1, problem.lower_bound, problem.upper_bound, **kw)
    opt = bea.BEATensor() if tensorized else bea.BEA()

    history = []

    class Track:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def on_iteration_end(self, iteration, runner):
            history.append(float(np.asarray(runner.space.best_agent.fit).reshape(-1)[0]))

    runner = Opytimizer(space, opt, Function(problem), enable_logging=False, show_progress=False)
    runner.start(MaxEvaluations(20000), callbacks=[Track()])

    assert all(b <= a for a, b in zip(history, history[1:]))
    assert history[-1] < 1e-6


def test_bea_counts_de_evaluations():
    np.random.seed(0)
    problem = Sphere(n_variables=3)
    space = SearchSpace(30, 3, 1, problem.lower_bound, problem.upper_bound)
    opt = bea.BEA(max_generations=10, de_start=0.5)
    runner = Opytimizer(space, opt, Function(problem), enable_logging=False, show_progress=False)
    runner.start(MaxIterations(10))

    # 10 generations of offspring plus 5 generations of DE trial vectors
    assert runner.function.n_calls == 10 * opt.n_offspring_per_generation + 5 * 30


def test_bea_gpu():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device available")

    cp.random.seed(0)
    env = Environment("cupy", "float32")
    problem = Rastrigin(n_variables=5, env=env)
    space = SearchSpace(100, 5, 1, problem.lower_bound, problem.upper_bound, env=env, tensorized=True)

    runner = Opytimizer(space, bea.BEATensor(max_generations=100), Function(problem), enable_logging=False, show_progress=False)
    runner.start(MaxIterations(100))

    assert float(np.asarray(runner.space.best_agent.fit).reshape(-1)[0]) < 5.0
