import numpy as np
import pytest
from opytimizer.core.agent import Agent
from opytimizer.core.space import _MultiObjectiveSpace
from opytimizer.optimizers.multi_objective.evolutionary  import SPEA2


@pytest.fixture
def dummy_space():
    """Creates a dummy multi-objective space with agents for testing."""
    space = _MultiObjectiveSpace(
        n_agents=4, n_variables=2, n_dimensions=1, n_objectives=2,
        lower_bound=[0.] * 2, upper_bound=[1.] * 2
    )

    space.build()

    # Set dummy objective values for dominance testing:
    # Agent 0: Dominating (1, 1)
    # Agent 1: Dominated by 0 (2, 2)
    # Agent 2: Dominated by 0 and 1 (3, 3)
    # Agent 3: Non-dominated trade-off (0.5, 4.0)
    space.agents[0].fit = np.array([1.0, 1.0])
    space.agents[1].fit = np.array([2.0, 2.0])
    space.agents[2].fit = np.array([3.0, 3.0])
    space.agents[3].fit = np.array([0.5, 4.0])
    return space


def test_spea2_initialization():
    """Tests SPEA2 optimizer initialization and property validation."""
    opt = SPEA2(params={"archive_size": 50})
    assert opt.archive_size == 50
    assert opt.archive == []

    with pytest.raises(Exception):
        SPEA2(params={"archive_size": -10})

    with pytest.raises(Exception):
        SPEA2(params={"archive_size": "invalid"})


def test_compile(dummy_space):
    """Tests the compile method of SPEA2."""
    opt = SPEA2()
    opt.compile(dummy_space)

    assert len(opt.archive) == 0
    assert len(opt.strength) == dummy_space.n_agents
    assert len(opt.raw_fitness) == dummy_space.n_agents
    assert len(opt.density) == dummy_space.n_agents


def test_metrics_calculation(dummy_space):
    """Tests Strength Eq. (1), Raw Fitness Eq. (2), and Density Eq. (3) calculations."""
    opt = SPEA2()
    opt._update_metrics(dummy_space.agents)

    # Agent 0 dominates 1 and 2 -> Strength should be 2
    assert opt.strength[0] == 2.0
    # Agent 1 dominates 2 -> Strength should be 1
    assert opt.strength[1] == 1.0
    # Agent 2 dominates nobody -> Strength should be 0
    assert opt.strength[2] == 0.0

    # Agent 0 is not dominated by anyone -> R(0) = 0
    assert opt.raw_fitness[0] == 0.0
    # Agent 1 is dominated by Agent 0 -> R(1) = S(0) = 2.0
    assert opt.raw_fitness[1] == 2.0
    # Agent 2 is dominated by 0 and 1 -> R(2) = S(0) + S(1) = 3.0
    assert opt.raw_fitness[2] == 3.0

    # Density D(i) should be < 0.5 for all agents (Eq. 3: 1 / (\sigma_k + 2))
    assert np.all(opt.density > 0)
    assert np.all(opt.density < 0.5)


def test_environmental_selection_truncation():
    """Tests truncation operator when the number of non-dominated agents exceeds archive_size."""
    opt = SPEA2(params={"archive_size": 2})

    agents = []
    # Create 3 non-dominated agents close to each other
    for fit_val in [[1.0, 5.0], [1.1, 4.9], [10.0, 1.0]]:
        a = Agent(n_variables=2, n_dimensions=1, n_objectives=2, lower_bound=[0.] * 2, upper_bound=[1.] * 2)
        a.fit = np.array(fit_val)
        agents.append(a)

    selected = opt._environmental_selection(agents, archive=[])
    assert len(selected) == 2


def test_update_cycle(dummy_space):
    """Tests a complete cycle of the update function."""
    opt = SPEA2(params={"archive_size": 2})
    opt.compile(dummy_space)

    def dummy_func(x):
        return np.array([np.sum(x), np.sum(x**2)])

    opt.update(dummy_space, dummy_func)

    # Check if the archive was updated according to archive_size
    assert len(opt.archive) == 2
    # Check if the population in space was updated with offspring
    assert len(dummy_space.agents) == 4