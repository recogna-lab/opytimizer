import numpy as np
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary import NSGA2
from opytimizer.visualization import plot_agents
from opytimizer import Opytimizer

def test_pareto_front_plot():
    func = Function(lambda x: np.array([x[0][0], 1-x[0][0]]))
    space = SearchSpace(10, 2, 2, [0.]*2, [1.]*2)
    opt = Opytimizer(space, NSGA2(), func)
    opt.start(MaxIterations(5))

    try:
        plot_agents(opt.space.pareto_front, labels=['F1', 'F2'], backend="matplotlib")
    except:
        plot_agents(opt.space.pareto_front, backend="matplotlib")