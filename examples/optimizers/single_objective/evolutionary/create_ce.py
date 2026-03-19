from opytimizer.optimizers.single_objective.evolutionary import CE

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "CR": 0.7,
    "DR": 0.7,
    "chaotic_system": 'gauss',
    "jump": 30
}

# Creates a CE optimizer
o = CE(params=params)
