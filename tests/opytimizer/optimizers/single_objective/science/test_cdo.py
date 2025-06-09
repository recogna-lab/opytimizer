import numpy as np

from opytimizer.optimizers.single_objective.science import cdo
from opytimizer.spaces import search


def test_cdo_update():
    def square(x):
        return np.sum(x**2)

    new_cdo = cdo.CDO()

    search_space = search.SearchSpace(
        n_agents=20,
        n_variables=2,
        n_objectives=1,
        lower_bound=[0, 0],
        upper_bound=[10, 10],
    )

    new_cdo.compile(search_space)

    new_cdo.update(search_space, square, iteration=1, n_iterations=20)
