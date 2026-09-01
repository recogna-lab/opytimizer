from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from opytimizer.core import Agent
import opytimizer.utils.exception as e

from opytimizer.visualization._core import fields as F
from opytimizer.visualization._core.transfer import batch_agents_to_matrices, to_numpy


def extract_data(
    result=None,
    *args: Union[List[Agent], Tuple[Any, Any], Any],
    fields: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict:
    if not args:
        raise e.ValueError("No Pareto fronts provided for comparison.")

    fmap = F.resolve(fields)
    out: Dict = {}

    need_matrix = F.wants_any(fmap, "fronts", "n_obj", "obj_labels")
    if need_matrix:
        is_agent_lists = (
            isinstance(args[0], list)
            and len(args[0]) > 0
            and isinstance(args[0][0], Agent)
        )

        matrices = []
        for arg in args:
            # Case 1: Agent list
            if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], Agent):
                matrices.append(batch_agents_to_matrices([arg])[0])

            # Case 2: Tuple
            elif isinstance(arg, tuple) and len(arg) == 2:
                matrices.append(to_numpy(arg[1]))

            # Case 3: n-dimensional array
            else:
                matrices.append(to_numpy(arg))
                
        for idx, matrix in enumerate(matrices):
            if matrix.ndim != 2:
                raise e.ValueError(
                    f"Front at index {idx} must be 2D with shape (N, M), got shape {matrix.shape}."
                )

        n_obj = matrices[0].shape[1]

        if n_obj not in (2, 3):
            raise e.ValueError(f"Comparison supports 2D or 3D, got {n_obj}D.")

        if F.wants(fmap, "fronts"):
            out["fronts"] = matrices
        if F.wants(fmap, "n_obj"):
            out["n_obj"] = n_obj
        if F.wants(fmap, "obj_labels"):
            out["obj_labels"] = (
                kwargs.get("obj_labels") or [f"Obj {i+1}" for i in range(n_obj)]
            )

    if F.wants(fmap, "title"):
        out["title"] = kwargs.get("title") or "Pareto Front Comparison"
    if F.wants(fmap, "alg_labels"):
        out["alg_labels"] = (
            kwargs.get("labels") or [f"Algorithm {i+1}" for i in range(len(args))]
        )
    if F.wants(fmap, "colors"):
        out["colors"] = plt.colormaps.get_cmap("tab10")(
            np.linspace(0, 1, len(args))
        )

    return out


def draw_mpl(ax, data: Dict) -> None:
    n_obj = data["n_obj"]

    if n_obj == 3 and ax.name != "3d":
        ax = ax.get_figure().add_subplot(111, projection="3d")

    for fit, name, color in zip(data["fronts"], data["alg_labels"], data["colors"]):
        if n_obj == 2:
            idx = np.argsort(fit[:, 0])
            ax.plot(fit[idx, 0], fit[idx, 1], "-", color=color, alpha=0.4)
            ax.scatter(
                fit[:, 0], fit[:, 1],
                color=color, label=name, edgecolors="k", s=35,
            )
        else:
            ax.scatter(
                fit[:, 0], fit[:, 1], fit[:, 2],
                color=color, label=name, s=30,
            )
            ax.set_zlabel(data["obj_labels"][2])

    ax.set_title(data["title"])
    ax.set_xlabel(data["obj_labels"][0])
    ax.set_ylabel(data["obj_labels"][1])
    ax.legend()


def draw_ply(fig, data: Dict) -> None:
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go

    for fit, name, color in zip(data["fronts"], data["alg_labels"], data["colors"]):
        color_hex = mcolors.to_hex(color)
        if data["n_obj"] == 2:
            idx = np.argsort(fit[:, 0])
            fig.add_trace(go.Scatter(
                x=fit[idx, 0], y=fit[idx, 1],
                mode="lines+markers",
                name=name,
                marker=dict(color=color_hex),
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=fit[:, 0], y=fit[:, 1], z=fit[:, 2],
                mode="markers",
                name=name,
                marker=dict(color=color_hex),
            ))

    fig.update_layout(title=data["title"])