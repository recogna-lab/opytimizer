---
sidebar_position: 19
title: Machine Learning Integrations
---

# Machine Learning Integrations

Learn how to fine-tune the hyperparameters of machine learning models, such as Scikit-Learn and PyTorch models, using Opytimizer.

## How It Works

The objective function receives the agent's position, with shape `(n_variables, n_dimensions)`, and returns a value to be minimized. Hence, each decision variable is a hyperparameter, which can be accessed as `x[i][0]`.

Since Opytimizer minimizes the objective, metrics that should be maximized (e.g., accuracy) must be converted, such as `1 - accuracy`.

## Example Code

```python
import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.swarm import PSO
from opytimizer.spaces import SearchSpace

# Loads digits dataset
digits = load_digits()

# Gathers samples and targets
X = digits.data
Y = digits.target


def _svm(x):
    # Gathers the hyperparameter
    C = x[0][0]

    # Instantiates an SVC class
    svc = svm.SVC(C=C, kernel="linear")

    # Fits the model using cross-validation
    scores = cross_val_score(svc, X, Y, cv=KFold(n_splits=5), n_jobs=-1)

    # Minimizes the error, i.e., 1 - mean accuracy
    return 1 - np.mean(scores)


# Number of agents and decision variables
n_agents = 10
n_variables = 1
n_objectives = 1

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [0.000001]
upper_bound = [10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, n_objectives, lower_bound, upper_bound)
optimizer = PSO()
function = Function(_svm)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function)

# Runs the optimization task
opt.start(MaxIterations(100))

# Prints out the best hyperparameter found
print(f"Best C: {opt.space.best_agent.position[0][0]} | Error: {opt.space.best_agent.fit}")
```

## Other Integrations

The `examples/integrations` folder of the repository contains additional examples:

* **`pytorch`**: Linear and logistic regression, neural networks, CNNs, and LSTMs.
* **`tensorflow`**: Convolutional neural networks.
* **`sklearn`**: Support Vector Machines and K-Means clustering.
* **`learnergy`**: Restricted Boltzmann Machines.
* **`nalp`**: Recurrent neural networks for natural language processing.
* **`opfython`**: Optimum-Path Forest feature selection and clustering.
