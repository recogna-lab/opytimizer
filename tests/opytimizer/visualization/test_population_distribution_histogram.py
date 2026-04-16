import numpy as np
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.visualization import population_distribution_histogram
from opytimizer import Opytimizer

def test_population_distribution_plot():
    func = Function(lambda x: np.array([x[0][0], 1-x[0][0]]))
    space = SearchSpace(10, 2, 2, [0.]*2, [1.]*2)
    opt = Opytimizer(space, NSGA2(), func)
    opt.start(5)

    try:
        population_distribution_histogram(opt.space.pareto_front, target=1, backend="matplotlib")
    except:
        population_distribution_histogram(opt.space.pareto_front, backend="matplotlib")