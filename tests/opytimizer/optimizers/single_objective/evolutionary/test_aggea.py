import numpy as np
import pytest

from opytimizer import Opytimizer
from opytimizer.benchmarks import Rastrigin, Sphere
from opytimizer.core import Environment, Function
from opytimizer.core.stopping import MaxEvaluations
from opytimizer.optimizers.single_objective.evolutionary import aggea
from opytimizer.spaces.search import SearchSpace
from opytimizer.utils import exception as e


def test_aggea_default_graph():
    opt = aggea.AGGEA()

    assert (opt.n_tournaments, opt.n_exchanges, opt.n_crossovers, opt.n_mutations) == (10, 10, 2, 2)
    assert np.all(opt.tournament_sizes == 2)
    assert np.allclose(opt.exchange_probabilities, 0.5)
    assert np.allclose(opt.crossover_split, [[0.4, 0.6], [0.4, 0.6]])
    assert np.allclose(opt.crossover_weights.sum(axis=1), 1.0)


def test_aggea_invalid_parameters():
    with pytest.raises(e.ValueError):
        aggea.AGGEA(tournament_sizes=[11] * 10)

    with pytest.raises(e.ValueError):
        aggea.AGGEA(n_tournaments=0)

    with pytest.raises(e.ValueError):
        aggea.AGGEA(exchange_probabilities=[-1.0, 1.0])

    with pytest.raises(e.ValueError):
        aggea.AGGEA(crossover_size=1)


@pytest.mark.parametrize("n_agents", [7, 50, 100, 333])
def test_aggea_graph_sizes_divide_exactly(n_agents):
    opt = aggea.AGGEA()
    n_parents, n_offspring = opt._graph_sizes(n_agents)

    assert n_offspring >= n_agents
    assert n_parents % opt.n_exchanges == 0
    assert (opt.n_tournaments * n_parents // opt.n_exchanges) % opt.n_crossovers == 0


def test_aggea_regroup_mixes_every_block():
    X = np.arange(2 * 4).reshape(2, 4, 1)  # 2 blocks of 4 solutions
    out = aggea._regroup(X, 2)

    # Each next block receives one half of every previous block
    assert out[:, :, 0].tolist() == [[0, 1, 4, 5], [2, 3, 6, 7]]


def test_aggea_categorical_follows_probabilities():
    np.random.seed(0)
    probabilities = np.array([[0.2, 0.8], [1.0, 0.0]])
    samples = aggea._categorical(probabilities, (2, 20000), np)

    assert samples[0].mean() == pytest.approx(0.8, abs=0.02)
    assert np.all(samples[1] == 0)


def test_aggea_offspring_shape_and_bounds():
    np.random.seed(0)
    opt = aggea.AGGEA()
    cfg = opt._prepare(100, 5, np, float)
    X = np.random.uniform(-1, 1, (100, 5))
    F = np.sum(X**2, axis=1)

    O = aggea._aggea_offspring(X, F, np.full(5, -1.0), np.full(5, 1.0), cfg, np)

    assert O.shape == (opt.n_offspring_per_generation, 5)
    assert np.all((O >= -1.0) & (O <= 1.0))


def test_aggea_exchange_only_recombines_parent_values():
    np.random.seed(0)
    # No crossover weights and no mutation: offspring variables are parent values
    opt = aggea.AGGEA(crossover_stds=0.0, crossover_means=0.0, mutation_stds=0.0)
    cfg = opt._prepare(40, 3, np, float)
    X = np.random.uniform(0, 1, (40, 3))

    O = aggea._aggea_offspring(X, X.sum(axis=1), np.zeros(3), np.ones(3), cfg, np)

    for k in range(3):
        assert np.all(np.isin(O[:, k], X[:, k]))


def test_aggea_tensor_matches_cpu_generation():
    opt = aggea.AGGEA()
    cfg = opt._prepare(100, 4, np, float)
    X = np.random.default_rng(1).uniform(-5, 5, (100, 4))
    F = np.sum(X**2, axis=1)
    lb, ub = np.full(4, -5.0), np.full(4, 5.0)

    np.random.seed(3)
    a = aggea._aggea_offspring(X, F, lb, ub, cfg, np)
    np.random.seed(3)
    b = aggea._aggea_offspring(X, F, lb, ub, cfg, np)

    assert np.array_equal(a, b)


@pytest.mark.parametrize("tensorized", [False, True])
def test_aggea_is_elitist_and_converges(tensorized):
    np.random.seed(0)
    env = Environment("numpy", "float64") if tensorized else None
    problem = Sphere(n_variables=5, env=env)
    kw = {"env": env, "tensorized": True} if tensorized else {}
    space = SearchSpace(50, 5, 1, problem.lower_bound, problem.upper_bound, **kw)
    opt = aggea.AGGEATensor() if tensorized else aggea.AGGEA()

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


def test_aggea_gpu():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device available")

    cp.random.seed(0)
    env = Environment("cupy", "float32")
    problem = Rastrigin(n_variables=5, env=env)
    space = SearchSpace(100, 5, 1, problem.lower_bound, problem.upper_bound, env=env, tensorized=True)

    runner = Opytimizer(space, aggea.AGGEATensor(), Function(problem), enable_logging=False, show_progress=False)
    runner.start(MaxEvaluations(20000))

    assert float(np.asarray(runner.space.best_agent.fit).reshape(-1)[0]) < 1.0
