from typing import List, Dict
import numpy as np

from opytimizer.core import Agent
import opytimizer.utils.exception as e

def extract_data(agents: List[Agent], *args, **kwargs) -> Dict:
    """
    Extracts fitness values from a list of Agent objects.
    Each agent must have an .fit attribute (numpy array).
    """
    # Convert list of agents to a 2D numpy array [n_agents, n_objectives]
    fitness_matrix = np.array([
    agent.fit.get() if hasattr(agent.fit, 'get') else agent.fit
    for agent in agents
    ])
    
    n_obj = fitness_matrix.shape[1]
    if n_obj not in [2, 3]:
        raise e.ValueError(f"Pareto front supports 2D or 3D fitness, got {n_obj}D.")

    labels = kwargs.get("labels") 
    if labels is None:
        labels = [f"Obj {i+1}" for i in range(n_obj)]
        
    return {
        "fitness": fitness_matrix,
        "n_obj": n_obj,
        "title": kwargs.get("title", "Pareto Front"),
        "color": kwargs.get("color", "#2ca02c"),
        "labels": labels
    }

def draw_mpl(ax, data):
    """Matplotlib implementation"""
    fit = data["fitness"]
    
    if data["n_obj"] == 2:
        ax.scatter(fit[:, 0], fit[:, 1], c=data["color"], edgecolors="k", zorder=3)
    else:
        # If 3-dimensional PF
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

def draw_ply(fig, data):
    """Plotly implementation (2D or 3D)"""
    import plotly.graph_objects as go
    fit = data["fitness"]

    if data["n_obj"] == 2:
        fig.add_trace(go.Scatter(
            x=fit[:, 0], y=fit[:, 1],
            mode="markers",
            marker=dict(color=data["color"], size=8, line=dict(width=1, color="black")),
            name="Pareto"
        ))
        fig.update_layout(xaxis_title=data["labels"][0], yaxis_title=data["labels"][1])
    else:
        fig.add_trace(go.Scatter3d(
            x=fit[:, 0], y=fit[:, 1], z=fit[:, 2],
            mode="markers",
            marker=dict(color=data["color"], size=5, line=dict(width=1, color="black")),
            name="Pareto"
        ))
        fig.update_layout(scene=dict(
            xaxis_title=data["labels"][0],
            yaxis_title=data["labels"][1],
            zaxis_title=data["labels"][2]
        ))

    fig.update_layout(title=data["title"])