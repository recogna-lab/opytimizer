"""
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from opytimizer.core import Agent
import opytimizer.utils.exception as e

from opytimizer.visualization._core import fields as F
from opytimizer.visualization._core.transfer import agents_to_matrix

def extract_data(
    agents: List[Agent],
    *args,
    fields: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict:
    """
    Extract agents data from a list of Agent objects.

    Parameters
    ----------
    agents : list of Agent with ``.fit`` attribute (CuPy or NumPy [n_obj])
    fields : iterable of field names to include, or ``None`` for all.
             Available fields: ``"fitness"``, ``"n_obj"``, ``"title"``,
             ``"color"``, ``"labels"``.

             The GPU → CPU transfer only happens when ``"fitness"`` (or a
             field that depends on it) is requested.

    **kwargs
        title  : str   – plot title            (default: "Pareto Front")
        color  : str   – marker colour         (default: "#2ca02c")
        labels : list  – axis labels           (default: ["Obj 1", ...])
    """
    fmap = F.resolve(fields)
    result: Dict = {}

    need_matrix = F.wants_any(fmap, "fitness", "n_obj", "labels")
    if need_matrix:
        fitness_matrix = agents_to_matrix(agents)   # single GPU → CPU hop
        n_obj = fitness_matrix.shape[1]

        if n_obj not in (2, 3):
            raise e.ValueError(
                f"This plot supports 2D or 3D fitness, got {n_obj}D."
            )

        if F.wants(fmap, "fitness"):
            result["fitness"] = fitness_matrix
        if F.wants(fmap, "n_obj"):
            result["n_obj"] = n_obj
        if F.wants(fmap, "labels"):
            result["labels"] = (
                kwargs.get("labels") or [f"Obj {i+1}" for i in range(n_obj)]
            )

    if F.wants(fmap, "title"):
        result["title"] = kwargs.get("title", "Solutions")
    if F.wants(fmap, "color"):
        result["color"] = kwargs.get("color", "#2ca02c")

    return result



def draw_mpl(ax, data: Dict) -> None:
    """Matplotlib: scatter plot of a 2-D or 3-D Agents."""
    fit = data["fitness"]

    if data["n_obj"] == 2:
        ax.scatter(fit[:, 0], fit[:, 1], c=data["color"], edgecolors="k", zorder=3)
    else:
        if ax.name != "3d":
            fig = ax.get_figure()
            ax.remove()
            ax = fig.add_subplot(111, projection="3d")

        ax.scatter(fit[:, 0], fit[:, 1], fit[:, 2], c=data["color"], edgecolors="k")
        ax.set_zlabel(data["labels"][2])

    ax.set_title(data["title"])
    ax.set_xlabel(data["labels"][0])
    ax.set_ylabel(data["labels"][1])
    ax.grid(True, linestyle="--", alpha=0.6)



def draw_ply(fig, data: Dict) -> None:
    """Plotly: interactive 2-D or 3-D Pareto front scatter."""
    import plotly.graph_objects as go

    fit = data["fitness"]
    marker = dict(color=data["color"], size=8, line=dict(width=1, color="black"))

    if data["n_obj"] == 2:
        fig.add_trace(go.Scatter(
            x=fit[:, 0], y=fit[:, 1],
            mode="markers",
            marker=marker,
            name="Feasible solution",
        ))
        fig.update_layout(
            xaxis_title=data["labels"][0],
            yaxis_title=data["labels"][1],
        )
    else:
        fig.add_trace(go.Scatter3d(
            x=fit[:, 0], y=fit[:, 1], z=fit[:, 2],
            mode="markers",
            marker=dict(color=data["color"], size=5, line=dict(width=1, color="black")),
            name="Feasible solution",
        ))
        fig.update_layout(scene=dict(
            xaxis_title=data["labels"][0],
            yaxis_title=data["labels"][1],
            zaxis_title=data["labels"][2],
        ))

    fig.update_layout(title=data["title"])