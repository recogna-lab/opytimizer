from opytimizer.optimizers.multi_objective.evolutionary import MOEAD

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"neighborhood_size": 20}

# Creates a MOEA/D optimizer
o = MOEAD(params=params)
