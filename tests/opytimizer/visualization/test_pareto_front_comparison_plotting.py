import numpy as np
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.multi_objective.evolutionary import RVEA, MOEAD
from opytimizer.visualization import plot_pareto_front_comparison
from opytimizer.utils.reference_vectors import das_dennis
from opytimizer import Opytimizer

def test_pareto_comparison_plot():
    def zdt1(x):
        x = x.flatten()
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1.0)
        return np.array([f1, g * (1.0 - np.sqrt(f1 / g))])

    W, N = das_dennis(2, 9)
    func = Function(zdt1)
    
    s1 = SearchSpace(N, 2, 2, [0.]*2, [1.]*2)
    o1 = Opytimizer(s1, RVEA(reference_vectors=W), func, save_agents=True)
    o1.start(MaxIterations(5))

    s2 = SearchSpace(N, 2, 2, [0.]*2, [1.]*2)
    o2 = Opytimizer(s2, MOEAD(weight_vectors=W), func, save_agents=True)
    o2.start(MaxIterations(5))

    try:
        plot_pareto_front_comparison(o1.space.pareto_front, o2.space.pareto_front, labels=['RVEA', 'MOEAD'], backend="matplotlib")
    except:
        plot_pareto_front_comparison(o1.space.pareto_front, o2.space.pareto_front, backend="matplotlib")