import numpy as np
import pytest
from opytimizer.core import agent
from opytimizer.core import space
from opytimizer.core.environment import Environment
import opytimizer.utils.exception as e

class ConcreteSpace(space._Space):
    def _create_agents(self) -> None:
        self.agents = [
            agent.Agent(
                n_variables=self.n_variables,
                n_dimensions=self.n_dimensions,
                n_objectives=self.n_objectives,
                lower_bound=self.lb,
                upper_bound=self.ub,
                mapping=self.mapping,
                env=self.env,
            )
            for _ in range(self.n_agents)
        ]



def test_space_n_agents():
    new_space = ConcreteSpace(n_agents=1)
    assert new_space.n_agents == 1


def test_space_n_agents_setter():
    new_space = ConcreteSpace(n_agents=1)

    with pytest.raises(e.TypeError):
        new_space.n_agents = 0.0

    with pytest.raises(e.ValueError):
        new_space.n_agents = 0

    new_space.n_agents = 5
    assert new_space.n_agents == 5


def test_space_n_variables():
    new_space = ConcreteSpace(n_variables=1)
    assert new_space.n_variables == 1


def test_space_n_variables_setter():
    new_space = ConcreteSpace(n_variables=1)

    with pytest.raises(e.TypeError):
        new_space.n_variables = 0.0

    with pytest.raises(e.ValueError):
        new_space.n_variables = 0

    new_space.n_variables = 2
    assert new_space.n_variables == 2


def test_space_n_dimensions():
    new_space = ConcreteSpace(n_dimensions=1)
    assert new_space.n_dimensions == 1


def test_space_n_dimensions_setter():
    new_space = ConcreteSpace(n_dimensions=1)

    with pytest.raises(e.TypeError):
        new_space.n_dimensions = 0.0

    with pytest.raises(e.ValueError):
        new_space.n_dimensions = 0

    new_space.n_dimensions = 3
    assert new_space.n_dimensions == 3


def test_space_n_objectives():
    new_space = ConcreteSpace(n_objectives=1)
    assert new_space.n_objectives == 1


def test_space_n_objectives_setter():
    new_space = ConcreteSpace(n_objectives=1)

    with pytest.raises(e.TypeError):
        new_space.n_objectives = 0.0

    with pytest.raises(e.ValueError):
        new_space.n_objectives = 0

    new_space.n_objectives = 2
    assert new_space.n_objectives == 2


def test_space_env():
    env = Environment().set_backend("numpy")
    new_space = ConcreteSpace(env=env)

    assert isinstance(new_space.env, Environment)

    with pytest.raises(e.TypeError):
        new_space.env = "numpy"


def test_space_agents():
    new_space = ConcreteSpace()
    assert new_space.agents == []


def test_space_agents_setter():
    new_space = ConcreteSpace()

    with pytest.raises(e.TypeError):
        new_space.agents = None

    new_space.agents = []
    assert new_space.agents == []


def test_space_lb():
    new_space = ConcreteSpace(n_variables=1)
    assert new_space.lb.shape == (1,)


def test_space_lb_setter():
    new_space = ConcreteSpace(n_variables=1)

    new_space.lb = np.array([1])
    assert new_space.lb[0] == 1

    with pytest.raises(e.SizeError):
        new_space.lb = np.array([1, 2])


def test_space_ub():
    new_space = ConcreteSpace(n_variables=1)
    assert new_space.ub.shape == (1,)


def test_space_ub_setter():
    new_space = ConcreteSpace(n_variables=1)

    new_space.ub = np.array([1])
    assert new_space.ub[0] == 1

    with pytest.raises(e.SizeError):
        new_space.ub = np.array([1, 2])


def test_space_mapping():
    new_space = ConcreteSpace(n_variables=1)
    assert len(new_space.mapping) == 1
    assert new_space.mapping == ["x0"]


def test_space_mapping_setter():
    new_space = ConcreteSpace(n_variables=1)

    with pytest.raises(e.TypeError):
        new_space.mapping = "a"

    with pytest.raises(e.SizeError):
        new_space.mapping = []

    new_space.mapping = ["x1"]
    assert len(new_space.mapping) == 1
    assert new_space.mapping == ["x1"]


def test_space_built():
    new_space = space._SingleObjectiveSpace()
    assert new_space.built is False


def test_space_built_setter():
    new_space = space._SingleObjectiveSpace()

    with pytest.raises(e.TypeError):
        new_space.built = 1

    new_space.built = True
    assert new_space.built is True


def test_single_objective_space_best_agent():
    new_space = space._SingleObjectiveSpace()
    assert isinstance(new_space.best_agent, agent.Agent)


def test_single_objective_space_best_agent_setter():
    new_space = space._SingleObjectiveSpace()

    with pytest.raises(e.TypeError):
        new_space.best_agent = None

    new_space.best_agent = agent.Agent(
        n_variables=1, n_dimensions=1, n_objectives=1, lower_bound=1, upper_bound=1
    )
    assert isinstance(new_space.best_agent, agent.Agent)


def test_single_objective_space_build():
    new_space = space._SingleObjectiveSpace(
        n_agents=2, n_variables=1, n_dimensions=1
    )
    new_space.build()

    assert len(new_space.agents) == 2
    assert new_space.built is True


def test_single_objective_space_clip_by_bound():
    new_space = space._SingleObjectiveSpace(
        n_agents=1, n_variables=1, lower_bound=0.0, upper_bound=1.0
    )
    new_space.build()
    new_space.agents[0].position[0] = -5.0
    new_space.clip_by_bound()

    assert new_space.agents[0].position[0] == 0.0


def test_multi_objective_space_pareto_front():
    new_space = space._MultiObjectiveSpace()
    assert new_space.pareto_front == []


def test_multi_objective_space_update_pareto_front():
    new_space = space._MultiObjectiveSpace(
        n_agents=2, n_variables=1, n_objectives=2
    )
    new_space.build()

    new_space.agents[0].fit = np.array([1.0, 2.0])
    new_space.agents[1].fit = np.array([2.0, 1.0])
    new_space.update_pareto_front()

    assert len(new_space.pareto_front) == 2



def test_single_objective_tensor_space_create_agents():
    new_space = space._SingleObjectiveTensorSpace(
        n_agents=2, n_variables=3, lower_bound=[0.0] * 3, upper_bound=[1.0] * 3
    )
    new_space._create_agents()

    assert new_space.X.shape == (2, 3)
    assert new_space.F.shape == (2,)
    assert len(new_space.agents) == 2


def test_single_objective_tensor_space_clip_by_bound():
    new_space = space._SingleObjectiveTensorSpace(
        n_agents=2, n_variables=2, lower_bound=[0.0]*2, upper_bound=[1.0]*2
    )
    new_space.build()
    new_space.X[0] = np.array([-1.0, 2.0])
    new_space.clip_by_bound()

    assert new_space.X[0, 0] == 0.0
    assert new_space.X[0, 1] == 1.0


def test_multi_objective_tensor_space_create_agents():
    new_space = space._MultiObjectiveTensorSpace(
        n_agents=2, n_variables=3, n_objectives=2, lower_bound=[0.0]*3, upper_bound=[1.0]*3
    )
    new_space._create_agents()

    assert new_space.X.shape == (2, 3)
    assert new_space.F.shape == (2, 2)


def test_multi_objective_tensor_space_update_pareto_front():
    new_space = space._MultiObjectiveTensorSpace(
        n_agents=2, n_variables=2, n_objectives=2, lower_bound=[0.0]*2, upper_bound=[1.0]*2
    )
    new_space.build()

    new_space.F = np.array([[1.0, 2.0], [2.0, 1.0]])
    new_space.X = np.array([[0.1, 0.2], [0.3, 0.4]])

    new_space.update_pareto_front()
    assert len(new_space.pareto_front[0]) == 2