"""
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from opytimizer.core import Agent

from opytimizer.visualization._core import fields as F
from opytimizer.visualization._core.transfer import agents_to_matrix


def extract_data(
    agents: List[Agent],
    *args,
    fields: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict:
    """
    Extract fitness distribution data for one objective.

    Parameters
    ----------
    agents : list of Agent with ``.fit`` attribute
    fields : iterable of field names, or ``None`` for all.
             Available: ``"fitness"``, ``"title"``, ``"color"``, ``"label"``.
    **kwargs
        target : int  – 1-based objective index to plot (default: 1)
        title  : str
        color  : str  – bar colour (default: "#2ca02c")
        label  : str  – x-axis label
    """
    fmap = F.resolve(fields)
    result: Dict = {}

    target_obj: int = kwargs.get("target", 1)

    if F.wants(fmap, "fitness"):
        matrix = agents_to_matrix(agents)          # [N, n_obj]
        result["fitness"] = matrix[:, target_obj - 1]

    if F.wants(fmap, "title"):
        result["title"] = kwargs.get("title") or f"Objective {target_obj} distribution"
    if F.wants(fmap, "color"):
        result["color"] = kwargs.get("color", "#2ca02c")
    if F.wants(fmap, "label"):
        result["label"] = kwargs.get("label") or "Count"

    return result



def draw_mpl(ax, data: Dict) -> None:
    """Matplotlib: histogram with Sturges bin count."""
    fit = data["fitness"]
    k = int(1 + 3.322 * np.log(len(fit)))   # Sturges' rule

    ax.hist(
        fit, bins=k,
        linewidth=0.5, edgecolor="black", color=data["color"],
    )
    ax.set_title(data["title"])
    ax.set_xlabel(data["label"])
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.6)



def draw_ply(fig, data: Dict) -> None:
    """Plotly: interactive histogram."""
    import plotly.graph_objects as go

    fig.add_trace(go.Histogram(
        x=data["fitness"],
        marker=dict(
            color=data["color"],
            line=dict(color="#000000", width=1),
        ),
    ))
    fig.update_layout(
        title=data["title"],
        xaxis_title=data["label"],
        yaxis_title="Count",
    )