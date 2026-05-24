import numpy as np
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace
from opytimizer.optimizers.single_objective.evolutionary import DE
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.optimizers.single_objective.population import GWO
from opytimizer.visualization import plot_convergence
from opytimizer import Opytimizer
from opytimark.markers.n_dimensional import Sphere

def test_convergence_plot():
    np.random.seed(0)
    N_AGENTS, N_GENERATIONS = 10, 10
    n_variables, n_objectives = 2, 1
    lb, ub = [-10, -10], [10, 10]
    func = Function(Sphere())

    def run_opt(opt_algo):
        space = SearchSpace(N_AGENTS, n_variables, n_objectives, lb, ub)
        opt = Opytimizer(space, opt_algo, func, save_agents=False)
        opt.start(N_GENERATIONS)
        return opt.history.best_agent

    h_de = run_opt(DE())
    h_pso = run_opt(PSO())
    h_gwo = run_opt(GWO())

    try:
        plot_convergence(h_de, h_pso, h_gwo, labels=['DE', 'PSO', 'GWO'], backend='matplotlib')
    except:
        plot_convergence(h_de, h_pso, h_gwo, backend='matplotlib')