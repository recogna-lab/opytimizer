import numpy as np
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.visualization import plot_pareto_front_evolution
from opytimizer import Opytimizer

def test_pareto_evolution_plot():
    func = Function(lambda x: np.array([x[0][0], 1-x[0][0]]))
    space = SearchSpace(10, 2, 2, [0.]*2, [1.]*2)
    opt = Opytimizer(space, NSGA2(), func)
    opt.start(10)

    try:
        plot_pareto_front_evolution(opt.history.pareto_front, iterations=[1, 5, 10], labels=['F1', 'F2'], backend="matplotlib")
    except:
        plot_pareto_front_evolution(opt.history.pareto_front, backend="matplotlib")