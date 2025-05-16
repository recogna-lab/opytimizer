from opytimizer.optimizers.single_objective.science import SA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"T": 100, "beta": 0.999}

# Creates a SA optimizer
o = SA(params=params)
