from opytimizer.optimizers.single_objective.science import ESA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"n_electrons": 5}

# Creates an ESA optimizer
o = ESA(params=params)
