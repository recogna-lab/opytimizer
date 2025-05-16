import numpy as np

from opytimizer.optimizers.single_objective.boolean import BMRFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"S": np.array([1])}

# Creates a BMRFO optimizer
o = BMRFO(params=params)
