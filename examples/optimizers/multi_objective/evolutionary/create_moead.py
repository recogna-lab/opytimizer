from opytimizer.optimizers.multi_objective.evolutionary import MOEAD

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"CR": 0.9, "MR": 0.05}

# Creates a PSO optimizer
o = MOEAD(params=params)