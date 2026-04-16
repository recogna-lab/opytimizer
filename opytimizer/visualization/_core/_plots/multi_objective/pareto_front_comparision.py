from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from opytimizer.core import Agent
import opytimizer.utils.exception as e

def extract_data(result = None, *args: List[List[Agent]], **kwargs) -> Dict:
    """
    Extracts final Pareto fronts for Multi-Objective (MO) algorithm comparison.
    Each arg is a list of Agents representing an algorithm's final front.
    """
    if not args:
        raise e.ValueError("No Pareto fronts provided for comparison.")

    # Convert agents to fitness matrices
    fronts = [np.array([ag.fit for ag in alg_pf]) for alg_pf in args]
    n_obj = fronts[0].shape[1]

    if n_obj not in [2, 3]:
        raise e.ValueError(f"Comparison supports 2D or 3D, got {n_obj}D.")

    return {
        "fronts": fronts,
        "n_obj": n_obj,
        "title": kwargs.get("title") or "Pareto Front Comparison",
        "alg_labels": kwargs.get("labels") or [f"Algorithm {i+1}" for i in range(len(args))],
        "obj_labels": kwargs.get("obj_labels") or [f"Obj {i+1}" for i in range(n_obj)],
        "colors": plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, len(args)))
    }

def draw_mpl(ax, data: Dict):
    """Matplotlib: Directly compares MO algorithm performance."""
    n_obj = data["n_obj"]
    
    if n_obj == 3 and ax.name != "3d":
        ax = ax.get_figure().add_subplot(111, projection="3d")

    for i, (fit, name) in enumerate(zip(data["fronts"], data["alg_labels"])):
        color = data["colors"][i]
        if n_obj == 2:
            # Sort for boundary visualization
            idx = np.argsort(fit[:, 0])
            ax.plot(fit[idx, 0], fit[idx, 1], "-", color=color, alpha=0.4)
            ax.scatter(fit[:, 0], fit[:, 1], color=color, label=name, edgecolors="k", s=35)
        else:
            ax.scatter(fit[:, 0], fit[:, 1], fit[:, 2], color=color, label=name, s=30)
            ax.set_zlabel(data["obj_labels"][2])

    ax.set_title(data["title"])
    ax.set_xlabel(data["obj_labels"][0])
    ax.set_ylabel(data["obj_labels"][1])
    ax.legend()

def draw_ply(fig, data: Dict):
    """Plotly: Interactive MO algorithm comparison."""
    import plotly.graph_objects as go
    import matplotlib.colors as mcolors

    for i, (fit, name) in enumerate(zip(data["fronts"], data["alg_labels"])):
        color_hex = mcolors.to_hex(data["colors"][i])
        if data["n_obj"] == 2:
            idx = np.argsort(fit[:, 0])
            fig.add_trace(go.Scatter(x=fit[idx, 0], y=fit[idx, 1], mode="lines+markers", 
                                     name=name, marker=dict(color=color_hex)))
        else:
            fig.add_trace(go.Scatter3d(x=fit[:, 0], y=fit[:, 1], z=fit[:, 2], 
                                       mode="markers", name=name, marker=dict(color=color_hex)))

    fig.update_layout(title=data["title"])