from opytimizer.optimizers.single_objective.evolutionary import OBCE

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "CR": 0.7,
    "DR": 0.7,
    "chaotic_system": 'gauss',
}

# Creates a OBCE optimizer
o = OBCE(params=params)
