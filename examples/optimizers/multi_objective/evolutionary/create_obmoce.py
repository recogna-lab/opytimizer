from opytimizer.optimizers.multi_objective.evolutionary import OBMOCE

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "CR": 0.7,
    "DR": 0.7,
    "chaotic_system": 'gauss',
}

# Creates a OBMOCE optimizer
o = OBMOCE(params=params)
