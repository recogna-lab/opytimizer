from opytimizer.optimizers.single_objective.swarm import BWO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"pp": 0.6, "cr": 0.44, "pm": 0.4}

# Creates a BWO optimizer
o = BWO(params=params)
