"""
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from opytimizer.core import Agent
import opytimizer.utils.exception as e

from opytimizer.visualization._core import fields as F
from opytimizer.visualization._core.transfer import batch_agents_to_matrices



def extract_data(
    pareto_fronts: List[List[Agent]],
    *args,
    fields: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict:
    """
    Extract selected Pareto-front snapshots for evolution plotting.

    Parameters
    ----------
    pareto_fronts : list of agent-lists, one entry per optimiser iteration.
    fields        : iterable of field names to include, or ``None`` for all.
                    Available: ``"evolution"``, ``"iterations"``, ``"n_obj"``,
                    ``"title"``, ``"labels"``, ``"cmap"``.
    **kwargs
        iterations : list[int] – 1-based iteration indices to show.
                                 ``None`` → show all.
        title      : str
        labels     : list[str] – axis labels
        cmap       : str       – matplotlib colormap name (default "viridis")
    """
    if not pareto_fronts:
        raise e.ValueError("pareto_fronts list is empty.")

    fmap = F.resolve(fields)
    result: Dict = {}

    requested_iters = kwargs.get("iterations")
    if requested_iters is not None:
        try:
            selected_groups = [pareto_fronts[i - 1] for i in requested_iters]
        except IndexError:
            raise e.ValueError(
                "One or more indices in 'iterations' are out of range."
            )
        iter_labels = requested_iters
    else:
        selected_groups = pareto_fronts
        iter_labels = list(range(1, len(pareto_fronts) + 1))

    
    need_matrix = F.wants_any(fmap, "evolution", "n_obj", "labels")
    if need_matrix:
        
        matrices = batch_agents_to_matrices(selected_groups)

        if not matrices or matrices[0].size == 0:
            raise e.ValueError("Could not extract fitness data from agents.")

        n_obj = matrices[0].shape[1]

        if F.wants(fmap, "evolution"):
            result["evolution"] = matrices
        if F.wants(fmap, "n_obj"):
            result["n_obj"] = n_obj
        if F.wants(fmap, "labels"):
            result["labels"] = (
                kwargs.get("labels") or [f"Obj {i+1}" for i in range(n_obj)]
            )

    
    if F.wants(fmap, "iterations"):
        result["iterations"] = iter_labels
    if F.wants(fmap, "title"):
        result["title"] = kwargs.get("title") or "Pareto Front Evolution"
    if F.wants(fmap, "cmap"):
        result["cmap"] = kwargs.get("cmap") or "viridis"

    return result


def draw_mpl(ax, data: Dict) -> None:
    """Matplotlib: temporal Pareto-front scatter with colour gradient."""
    import matplotlib.pyplot as plt

    n_obj = data["n_obj"]
    iters_data: List[np.ndarray] = data["evolution"]
    it_numbers: List[int] = data["iterations"]

    colors = plt.colormaps.get_cmap(data["cmap"])(
        np.linspace(0, 1, len(iters_data))
    )

    if n_obj == 3 and ax.name != "3d":
        fig = ax.get_figure()
        ax.remove()
        ax = fig.add_subplot(111, projection="3d")

    for color, fit, it_num in zip(colors, iters_data, it_numbers):
        label = f"Iter {it_num}"
        if n_obj == 2:
            idx = np.argsort(fit[:, 0])
            ax.plot(fit[idx, 0], fit[idx, 1], "--", color=color, alpha=0.4)
            ax.scatter(
                fit[:, 0], fit[:, 1],
                color=color, edgecolors="k",
                linewidth=0.5, s=30, label=label, zorder=3,
            )
        else:
            ax.scatter(
                fit[:, 0], fit[:, 1], fit[:, 2],
                color=color, s=30, alpha=0.7, label=label,
            )
            ax.set_zlabel(data["labels"][2])

    ax.set_title(data["title"], loc="left")
    ax.set_xlabel(data["labels"][0])
    ax.set_ylabel(data["labels"][1])
    ax.legend(fontsize="small", loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)



def draw_ply(fig, data: Dict) -> None:
    """Plotly: interactive temporal Pareto-front evolution."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go

    n_obj = data["n_obj"]
    iters_data: List[np.ndarray] = data["evolution"]
    it_numbers: List[int] = data["iterations"]

    cmap = plt.colormaps.get_cmap(data["cmap"])
    hex_colors = [
        mcolors.to_hex(cmap(t))
        for t in np.linspace(0, 1, len(iters_data))
    ]

    for color, fit, it_num in zip(hex_colors, iters_data, it_numbers):
        name = f"Iter {it_num}"
        if n_obj == 2:
            idx = np.argsort(fit[:, 0])
            fig.add_trace(go.Scatter(
                x=fit[idx, 0], y=fit[idx, 1],
                mode="lines+markers",
                name=name,
                line=dict(color=color, dash="dot", width=1.5),
                marker=dict(color=color, size=8, line=dict(width=1, color="black")),
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=fit[:, 0], y=fit[:, 1], z=fit[:, 2],
                mode="markers",
                name=name,
                marker=dict(color=color, size=5, opacity=0.8),
            ))

    fig.update_layout(
        title=data["title"],
        xaxis_title=data["labels"][0],
        yaxis_title=data["labels"][1],
        legend_title="Selected Iterations",
    )
    if n_obj == 3:
        fig.update_layout(scene=dict(
            xaxis_title=data["labels"][0],
            yaxis_title=data["labels"][1],
            zaxis_title=data["labels"][2],
        ))