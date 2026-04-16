from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import opytimizer.utils.exception as e
from opytimizer.core import Agent

def extract_data(Pareto_Fronts: List[List[Agent]], *args, **kwargs) -> Dict:
    """
    Extracts specific Pareto fronts based on provided iteration indices.
    """
    if not Pareto_Fronts:
        raise e.ValueError("Pareto_Fronts list is empty.")

    # Get requested indices from kwargs
    requested_iters = kwargs.get("iterations")

    if requested_iters is not None:
        # Extract ONLY the specific indices requested by the user
        try:
            pareto_evolution = [
                np.array([ag.fit for ag in Pareto_Fronts[i - 1]]) 
                for i in requested_iters
            ]
            iterations = requested_iters
        except IndexError:
            raise e.ValueError("One or more indices in 'iterations' are out of range.")
    else:
        # Default: extract all iterations
        pareto_evolution = [np.array([ag.fit for ag in pf]) for pf in Pareto_Fronts]
        iterations = list(range(len(pareto_evolution)))

    n_obj = pareto_evolution[0].shape[1]
    
    # Secure parameter retrieval (handling None or missing keys)
    labels = kwargs.get("labels") or [f"Obj {i+1}" for i in range(n_obj)]
    title = kwargs.get("title") or "Pareto Front Evolution"
    cmap = kwargs.get("cmap") or "viridis"

    return {
        "evolution": pareto_evolution,
        "iterations": iterations,
        "n_obj": n_obj,
        "title": title,
        "labels": labels,
        "cmap": cmap
    }

def draw_mpl(ax, data: Dict):
    """
    Plots precisely the extracted iterations using a temporal color gradient.
    """
    n_obj = data["n_obj"]
    iters_data = data["evolution"]
    it_numbers = data["iterations"]
    
    # Maps colors only to the selected snapshots
    colors = plt.colormaps.get_cmap(data['cmap'])(np.linspace(0, 1, len(iters_data)))

    # Handle 3D projection transition
    if n_obj == 3 and ax.name != "3d":
        fig = ax.get_figure()
        ax.remove()
        ax = fig.add_subplot(111, projection="3d")

    for i, (fit, it_num) in enumerate(zip(iters_data, it_numbers)):
        label = f"Iter {it_num}"
        
        if n_obj == 2:
            idx = np.argsort(fit[:, 0])
            ax.plot(fit[idx, 0], fit[idx, 1], "--", color=colors[i], alpha=0.4)
            ax.scatter(fit[:, 0], fit[:, 1], color=colors[i], edgecolors="k", 
                       linewidth=0.5, s=30, label=label, zorder=3)
        else:
            ax.scatter(fit[:, 0], fit[:, 1], fit[:, 2], color=colors[i], 
                       s=30, alpha=0.7, label=label)
            ax.set_zlabel(data["labels"][2])

    ax.set_title(data["title"], loc="left")
    ax.set_xlabel(data["labels"][0])
    ax.set_ylabel(data["labels"][1])
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, linestyle="--", alpha=0.5)
    
    
    
def draw_ply(fig, data: Dict):
    """
    Interactive Plotly rendering for the selected Pareto iterations.
    """
    import plotly.graph_objects as go
    import matplotlib.colors as mcolors
    
    n_obj = data["n_obj"]
    iters_data = data["evolution"]
    it_numbers = data["iterations"]
    cmap = plt.cm.get_cmap(data["cmap"])
    colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, len(iters_data))]

    for i, (fit, it_num) in enumerate(zip(iters_data, it_numbers)):
        name = f"Iter {it_num}"
        
        if n_obj == 2:
            idx = np.argsort(fit[:, 0])
            fig.add_trace(go.Scatter(
                x=fit[idx, 0], y=fit[idx, 1],
                mode="lines+markers",
                name=name,
                line=dict(color=colors[i], dash="dot", width=1.5),
                marker=dict(color=colors[i], size=8, line=dict(width=1, color="black"))
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=fit[:, 0], y=fit[:, 1], z=fit[:, 2],
                mode="markers",
                name=name,
                marker=dict(color=colors[i], size=5, opacity=0.8)
            ))

    fig.update_layout(
        title=data["title"],
        xaxis_title=data["labels"][0],
        yaxis_title=data["labels"][1],
        legend_title="Selected Iterations"
    )

    if n_obj == 3:
        fig.update_layout(scene=dict(
            xaxis_title=data["labels"][0],
            yaxis_title=data["labels"][1],
            zaxis_title=data["labels"][2]
        ))