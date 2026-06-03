"""
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import opytimizer.utils.exception as e

from opytimizer.visualization._core import fields as F



def extract_data(
    result=None,
    *args: List,
    fields: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict:
    """
    Extract convergence curves from one or more optimiser histories.

    Parameters
    ----------
    *args  : each positional arg is a convergence history –
             a list of ``(iteration, best_fitness)`` pairs.
    fields : iterable of field names, or ``None`` for all.
             Available: ``"curves"``, ``"x_axis"``, ``"labels"``,
             ``"title"``, ``"xlabel"``, ``"ylabel"``.
    **kwargs
        iterations : list[int] – 1-based iteration indices to extract.
                                 ``None`` → all iterations.
        labels     : list[str] – algorithm names
        title      : str
        xlabel     : str
        ylabel     : str
    """
    if not args:
        raise e.ValueError("No convergence data provided.")

    fmap = F.resolve(fields)
    out: Dict = {}

    need_curves = F.wants_any(fmap, "curves", "x_axis")
    if need_curves:
        requested_indices: Optional[List[int]] = kwargs.get("iterations")

        curves: List[List[float]] = []
        if requested_indices is not None:
            for opt in args:
                curves.append([
                    opt[i - 1][1]
                    for i in requested_indices
                    if i <= len(opt)
                ])
            x_axis = requested_indices
        else:
            for opt in args:
                curves.append([float(opt[i][1]) for i in range(len(opt))])
            x_axis = list(range(1, len(args[0]) + 1))

        if F.wants(fmap, "curves"):
            out["curves"] = curves
        if F.wants(fmap, "x_axis"):
            out["x_axis"] = x_axis

    if F.wants(fmap, "labels"):
        out["labels"] = (
            kwargs.get("labels") or [f"Alg {i+1}" for i in range(len(args))]
        )
    if F.wants(fmap, "title"):
        out["title"] = kwargs.get("title") or "Convergence Analysis"
    if F.wants(fmap, "xlabel"):
        out["xlabel"] = kwargs.get("xlabel") or "Iteration"
    if F.wants(fmap, "ylabel"):
        out["ylabel"] = kwargs.get("ylabel") or "Best Fitness"

    return out


def draw_mpl(ax, data: Dict) -> None:
    """Matplotlib: one line per algorithm."""
    for curve, label in zip(data["curves"], data["labels"]):
        ax.plot(
            data["x_axis"], curve,
            label=label,
            marker="o" if len(curve) < 20 else None,
        )

    ax.set_title(data["title"])
    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)


def draw_ply(fig, data: Dict) -> None:
    """Plotly: interactive convergence lines."""
    import plotly.graph_objects as go

    for curve, label in zip(data["curves"], data["labels"]):
        fig.add_trace(go.Scatter(
            x=data["x_axis"], y=curve,
            mode="lines",
            name=label,
        ))

    fig.update_layout(
        title=data["title"],
        xaxis_title=data["xlabel"],
        yaxis_title=data["ylabel"],
    )