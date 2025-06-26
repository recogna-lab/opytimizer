"""Selection-based hyperheuristics package.

This package contains hyperheuristics that select between different
low-level optimizers using various strategies.
"""

from opytimizer.hyperheuristics.selection.selection_hyperheuristic import (
    SelectionHyperHeuristic,
)

__all__ = ["SelectionHyperHeuristic"]
