from opytimizer.optimizers.single_objective.science import GSA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"G": 2.467}

# Creates a GSA optimizer
o = GSA(params=params)
