import opfython.math.general as g
import opfython.stream.splitter as s
from opfython.models.supervised import SupervisedOPF
from sklearn.datasets import load_digits

import opytimizer.math.random as r
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.boolean import BPSO
from opytimizer.spaces import BooleanSpace

# Loads digits dataset
digits = load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target

# Adding 1 to labels, i.e., OPF should have labels from 1+
Y += 1

# Splits data into training and testing sets
X_train, X_val, Y_train, Y_val = s.split(X, Y, percentage=0.5, random_state=1)


def supervised_opf_feature_selection(opytimizer):
    # Gathers features
    features = opytimizer[:, 0].astype(bool)

    # Remaking training and validation subgraphs with selected features
    X_train_selected = X_train[:, features]
    X_val_selected = X_val[:, features]

    # Creates a SupervisedOPF instance
    opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

    # Fits training data into the classifier
    opf.fit(X_train_selected, Y_train)

    # Predicts new data
    preds = opf.predict(X_val_selected)

    # Calculates accuracy
    acc = g.opf_accuracy(Y_val, preds)

    return 1 - acc


# Number of agents and decision variables
n_agents = 5
n_variables = 64

# Parameters for the optimizer
params = {
    "c1": r.generate_binary_random_number(size=(n_variables, 1)),
    "c2": r.generate_binary_random_number(size=(n_variables, 1)),
}

# Creates the space, optimizer and function
space = BooleanSpace(n_agents, n_variables)
optimizer = BPSO()
function = Function(supervised_opf_feature_selection)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(n_iterations=3)
