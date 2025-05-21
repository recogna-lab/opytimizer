from opytimizer.optimizers.multi_objective.evolutionary import NSGA2

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"crossover_rate": 0.9, "mutation_rate": 0.025}

# Creates a PSO optimizer
o = NSGA2(params=params)