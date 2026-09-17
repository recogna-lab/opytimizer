import pytest

from opytimizer.spaces.tree import (
    TreeSpace,
    _MultiObjectiveTreeSpace,
    _SingleObjectiveTreeSpace,
)


@pytest.mark.parametrize(
    ("n_objectives", "expected"),
    [
        (1, _SingleObjectiveTreeSpace),
        (2, _MultiObjectiveTreeSpace),
    ],
)
def test_tree_space_factory(n_objectives, expected, primitive_set):
    space = TreeSpace(
        n_agents=3,
        n_objectives=n_objectives,
        pset=primitive_set,
        min_depth=1,
        max_depth=2,
    )

    assert isinstance(space, expected)
    assert space.built is True
    assert len(space.agents) == 3
    assert all(agent.position is not None for agent in space.agents)


def test_tree_space_rejects_non_positive_objectives(primitive_set):
    with pytest.raises(Exception):
        TreeSpace(2, 0, primitive_set)



def test_single_objective_tree_space_copies_best_agent(primitive_set):
    space = TreeSpace(
        3,
        1,
        primitive_set,
        min_depth=1,
        max_depth=2,
    )

    assert space.best_agent.position is not space.agents[0].position
    assert space.best_agent.fit == space.agents[0].fit
