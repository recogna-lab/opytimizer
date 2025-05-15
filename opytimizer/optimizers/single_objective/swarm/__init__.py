"""An evolutionary package for all common opytimizer modules.
It contains implementations of swarm-based optimizers.
"""

from opytimizer.optimizers.single_objective.swarm.abc import ABC
from opytimizer.optimizers.single_objective.swarm.abo import ABO
from opytimizer.optimizers.single_objective.swarm.af import AF
from opytimizer.optimizers.single_objective.swarm.ba import BA
from opytimizer.optimizers.single_objective.swarm.boa import BOA
from opytimizer.optimizers.single_objective.swarm.bwo import BWO
from opytimizer.optimizers.single_objective.swarm.cs import CS
from opytimizer.optimizers.single_objective.swarm.csa import CSA
from opytimizer.optimizers.single_objective.swarm.eho import EHO
from opytimizer.optimizers.single_objective.swarm.fa import FA
from opytimizer.optimizers.single_objective.swarm.ffoa import FFOA
from opytimizer.optimizers.single_objective.swarm.fpa import FPA
from opytimizer.optimizers.single_objective.swarm.fso import FSO
from opytimizer.optimizers.single_objective.swarm.goa import GOA
from opytimizer.optimizers.single_objective.swarm.js import JS, NBJS
from opytimizer.optimizers.single_objective.swarm.kh import KH
from opytimizer.optimizers.single_objective.swarm.mfo import MFO
from opytimizer.optimizers.single_objective.swarm.mrfo import MRFO
from opytimizer.optimizers.single_objective.swarm.pio import PIO
from opytimizer.optimizers.single_objective.swarm.pso import AIWPSO, PSO, RPSO, SAVPSO, VPSO
from opytimizer.optimizers.single_objective.swarm.sbo import SBO
from opytimizer.optimizers.single_objective.swarm.sca import SCA
from opytimizer.optimizers.single_objective.swarm.sfo import SFO
from opytimizer.optimizers.single_objective.swarm.sos import SOS
from opytimizer.optimizers.single_objective.swarm.ssa import SSA
from opytimizer.optimizers.single_objective.swarm.sso import SSO
from opytimizer.optimizers.single_objective.swarm.stoa import STOA
from opytimizer.optimizers.single_objective.swarm.waoa import WAOA
from opytimizer.optimizers.single_objective.swarm.woa import WOA
