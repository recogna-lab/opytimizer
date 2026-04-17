"""An evolutionary package for all common opytimizer modules.
It contains implementations of evolutionary-based optimizers.
"""

from opytimizer.optimizers.multi_objective.evolutionary.moead import MOEAD, MOEAD_DE
from opytimizer.optimizers.multi_objective.evolutionary.nsga2 import NSGA2
from opytimizer.optimizers.multi_objective.evolutionary.spea2 import SPEA2
from opytimizer.optimizers.multi_objective.evolutionary.moce import MOCE, OBMOCE
from opytimizer.optimizers.multi_objective.evolutionary.rvea import RVEA
from opytimizer.optimizers.multi_objective.evolutionary.nsga3 import NSGA3
from opytimizer.optimizers.multi_objective.evolutionary.knea import KnEA
