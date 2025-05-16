from opytimizer.spaces import search


def test_search_initialize_agents():
    new_search_space = search.SearchSpace(1, 1, 0, 1)

    assert new_search_space.agents[0].position[0] != 0


def test_search_clip_by_bound():
    new_search_space = search.SearchSpace(1, 1, 0, 1)

    new_search_space.agents[0].position[0] = 20

    new_search_space.clip_by_bound()

    assert new_search_space.agents[0].position[0] != 20
