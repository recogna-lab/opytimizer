"""Hyperheuristics package for opytimizer.

This package contains various hyperheuristic implementations organized by category:
- selection: Selection-based hyperheuristics
- generation: Generation-based hyperheuristics
- adaptive: Adaptive hyperheuristics
- hybrid: Hybrid hyperheuristics combining multiple approaches
"""

from opytimizer.core.hyperheuristic import HyperHeuristic
from opytimizer.hyperheuristics.adaptive import AdaptiveHyperHeuristic
from opytimizer.hyperheuristics.hybrid import HybridHyperHeuristic

# Import specific hyperheuristic implementations
from opytimizer.hyperheuristics.selection import SelectionHyperHeuristic
from opytimizer.utils.adaptation_mechanism import (
    AdaptationMechanism,
    ParameterAdaptation,
    PopulationAdaptation,
    StrategyAdaptation,
)
from opytimizer.utils.performance_tracker import (
    MultiObjectivePerformanceTracker,
    PerformanceTracker,
)
from opytimizer.utils.selection_strategy import (
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
    "SelectionStrategy",
    "ChoiceFunction",
    "GreedySelection",
    "MultiArmedBandit",
    "RandomDescent",
    "SimulatedAnnealingSelection",
    "AdaptationMechanism",
    "ParameterAdaptation",
    "StrategyAdaptation",
    "PopulationAdaptation",
    "PerformanceTracker",
    "MultiObjectivePerformanceTracker",
]
