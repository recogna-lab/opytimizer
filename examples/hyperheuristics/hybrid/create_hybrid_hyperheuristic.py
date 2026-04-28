from opytimizer.hyperheuristics.adaptation_mechanism import StrategyAdaptation
from opytimizer.hyperheuristics.hybrid import HybridHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import MultiArmedBandit
from opytimizer.optimizers.single_objective.swarm import ABC, PSO

# One should declare a list of optimizer instances to be used by the hyperheuristic
optimizers = [
    PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}),
    ABC(params={"n_trials": 10}),
]

# One should declare the selection strategy and adaptation mechanism
selection_strategy = MultiArmedBandit()
adaptation_mechanism = StrategyAdaptation()

# Creates a HybridHyperHeuristic
h = HybridHyperHeuristic(
    optimizers=optimizers,
    selection_strategy=selection_strategy,
    adaptation_mechanism=adaptation_mechanism,
    selection_interval=10,
    adaptation_interval=20,
)
