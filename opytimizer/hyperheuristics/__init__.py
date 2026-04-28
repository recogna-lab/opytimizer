"""A hyperheuristics package for all common opytimizer modules.
It contains specific packages of every hyperheuristic taxonomy
covered by opytimizer.
"""

from opytimizer.core.hyperheuristic import HyperHeuristic
from opytimizer.hyperheuristics.adaptation_mechanism import (
    AdaptationMechanism,
    AdaptiveParameterControl,
    ParameterAdaptation,
    PopulationAdaptation,
    SelfAdaptiveControl,
    StrategyAdaptation,
)
from opytimizer.hyperheuristics.adaptive import AdaptiveHyperHeuristic
from opytimizer.hyperheuristics.generation import ComponentBasedHyperHeuristic
from opytimizer.hyperheuristics.hybrid import HybridHyperHeuristic
from opytimizer.hyperheuristics.selection import SelectionHyperHeuristic
from opytimizer.hyperheuristics.selection_strategy import (
    ChoiceFunction,
    GreedySelection,
    MultiArmedBandit,
    RandomDescent,
    SelectionStrategy,
    SimulatedAnnealingSelection,
)

__all__ = [
    "HyperHeuristic",
    "SelectionHyperHeuristic",
    "AdaptiveHyperHeuristic",
    "HybridHyperHeuristic",
    "ComponentBasedHyperHeuristic",
    "SelectionStrategy",
    "ChoiceFunction",
    "MultiArmedBandit",
    "RandomDescent",
    "GreedySelection",
    "SimulatedAnnealingSelection",
    "AdaptationMechanism",
    "ParameterAdaptation",
    "StrategyAdaptation",
    "PopulationAdaptation",
    "AdaptiveParameterControl",
    "SelfAdaptiveControl",
]
