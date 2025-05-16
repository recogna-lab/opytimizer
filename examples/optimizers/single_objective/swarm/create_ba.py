from opytimizer.optimizers.single_objective.swarm import BA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"f_min": 0, "f_max": 2, "A": 0.5, "r": 0.5}

# Creates a BA optimizer
o = BA(params=params)
