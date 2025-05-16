from opytimizer.optimizers.single_objective.swarm import KH

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "N_max": 0.01,
    "w_n": 0.42,
    "NN": 5,
    "V_f": 0.02,
    "w_f": 0.38,
    "D_max": 0.002,
    "C_t": 0.5,
    "Cr": 0.2,
    "Mu": 0.05,
}

# Creates a KH optimizer
o = KH(params=params)
