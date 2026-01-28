from opytimizer.optimizers.multi_objective.evolutionary import SPEA2

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"archive_size": 100}
# Creates a NSGA2 optimizer
o = SPEA2()
