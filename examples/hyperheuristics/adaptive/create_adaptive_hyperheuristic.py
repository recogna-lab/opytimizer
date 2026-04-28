from opytimizer.hyperheuristics.adaptation_mechanism import ParameterAdaptation
from opytimizer.hyperheuristics.adaptive import AdaptiveHyperHeuristic
from opytimizer.optimizers.single_objective.swarm import ABC, PSO

# One should declare a list of optimizer instances to be used by the hyperheuristic
optimizers = [
    PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}),
    ABC(params={"n_trials": 10}),
]

# One should declare the adaptation mechanism
adaptation_mechanism = ParameterAdaptation()

# Creates an AdaptiveHyperHeuristic
h = AdaptiveHyperHeuristic(
    optimizers=optimizers,
    adaptation_mechanism=adaptation_mechanism,
    adaptation_interval=20,
)
