import random

import numpy as np

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.core.stopping import MaxIterations
from opytimizer.optimizers.single_objective.evolutionary.gp import GP
from opytimizer.spaces.tree import TreeSpace
from opytimizer.visualization import plot_graph


class ArrayType:
    pass


SEED = 100
np.random.seed(SEED)
random.seed(SEED)


def protected_exp(a):
    return np.exp(np.clip(a, -10, 10))


def protected_log(a):
    return np.log(np.abs(a) + 1e-5)


def custom_if_gt0(cond, val_if_pos, val_if_neg):
    return np.where(cond > 0, val_if_pos, val_if_neg)


X_data = np.linspace(-3, 3, 100)
Y_target = np.cos(X_data) + protected_exp(-(X_data**2))

pset = PrimitiveSet(name="TrigRegression", root_type=ArrayType)


pset.add_primitive(np.sin, (ArrayType,), ArrayType, name="sin")
pset.add_primitive(protected_exp, (ArrayType,), ArrayType, name="exp")
pset.add_primitive(protected_log, (ArrayType,), ArrayType, name="log")


pset.add_primitive(np.add, (ArrayType, ArrayType), ArrayType, name="add")
pset.add_primitive(np.multiply, (ArrayType, ArrayType), ArrayType, name="mul")


pset.add_primitive(
    custom_if_gt0, (ArrayType, ArrayType, ArrayType), ArrayType, name="if_gt0"
)


pset.add_terminal(value=X_data, output_type=ArrayType, name="X")
pset.add_terminal(value=np.ones_like(X_data), output_type=ArrayType, name="1.0")

pset.validate()


def evaluate_trig(tree):
    try:
        y_pred = tree.evaluate()

        mae = np.mean(np.abs(y_pred - Y_target))
        penalty = tree.depth * 0.01
        return mae + penalty
    except Exception:
        return np.inf


function = Function(evaluate_trig)

space = TreeSpace(
    n_agents=10,
    n_objectives=1,
    pset=pset,
    min_depth=2,
    max_depth=4,
    method="half_and_half",
)

optimizer = GP()


opt = Opytimizer(space=space, function=function, optimizer=optimizer)

opt.start(MaxIterations(1000))

print(opt.space.best_agent.position)

plot_graph(opt.space.best_agent.position).show()
