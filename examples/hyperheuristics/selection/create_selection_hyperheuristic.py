from opytimizer.hyperheuristics.selection import SelectionHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import ChoiceFunction
from opytimizer.optimizers.single_objective.swarm import ABC, PSO

# One should declare a list of optimizer instances to be used by the hyperheuristic
optimizers = [
    PSO(params={"w": 0.7, "c1": 1.7, "c2": 1.7}),
    ABC(params={"n_trials": 10}),
]

# One should declare the selection strategy
selection_strategy = ChoiceFunction()

# Creates a SelectionHyperHeuristic
h = SelectionHyperHeuristic(
    optimizers=optimizers, selection_strategy=selection_strategy, selection_interval=10
)
