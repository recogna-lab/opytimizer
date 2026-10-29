"""Multi- and many-objective benchmark problems (ZDT, DTLZ, WFG and ZCAT).

Every problem is a callable that can be wrapped by `opytimizer.core.Function`,
runs on the numpy / cupy backend of its `env` and exposes `pareto_front()`.
"""

from opytimizer.benchmarks.multi_objective.dtlz import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
)
from opytimizer.benchmarks.multi_objective.wfg import (
    WFG1,
    WFG2,
    WFG3,
    WFG4,
    WFG5,
    WFG6,
    WFG7,
    WFG8,
    WFG9,
)
from opytimizer.benchmarks.multi_objective.zcat import (
    ZCAT1,
    ZCAT2,
    ZCAT3,
    ZCAT4,
    ZCAT5,
    ZCAT6,
    ZCAT7,
    ZCAT8,
    ZCAT9,
    ZCAT10,
    ZCAT11,
    ZCAT12,
    ZCAT13,
    ZCAT14,
    ZCAT15,
    ZCAT16,
    ZCAT17,
    ZCAT18,
    ZCAT19,
    ZCAT20,
)
from opytimizer.benchmarks.multi_objective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

__all__ = [
    "ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
    "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7",
    "WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9",
    "ZCAT1", "ZCAT2", "ZCAT3", "ZCAT4", "ZCAT5", "ZCAT6", "ZCAT7", "ZCAT8",
    "ZCAT9", "ZCAT10", "ZCAT11", "ZCAT12", "ZCAT13", "ZCAT14", "ZCAT15",
    "ZCAT16", "ZCAT17", "ZCAT18", "ZCAT19", "ZCAT20",
]  # fmt: skip
