from typing import List, Dict
import numpy as np

from opytimizer.core import Agent
import opytimizer.utils.exception as e

def extract_data(agents: List[Agent], *args, **kwargs) -> Dict:
    """
    Extracts fitness values from a list of Agent objects.
    """
    # Convert list of agents to a 2D numpy array [n_agents, n_objectives]
    fitness_matrix = np.array([
    agent.fit.get() if hasattr(agent.fit, 'get') else agent.fit
    for agent in agents
    ])
    
    target_obj = kwargs.get('target', 1)
    
    filtered_fitness = fitness_matrix[:, target_obj - 1]

    label = kwargs.get("label")
    title = kwargs.get("title") 
    if label is None:
        label = f"Count"
    if title is None:
        title = f"Objective {target_obj} distribution"
    return {
        "fitness": filtered_fitness,
        "title": title,
        "color": kwargs.get("color", "#2ca02c"),
        "label": label
    }

def draw_mpl(ax, data):
    """Matplotlib implementation"""
    fit = data["fitness"]
    
    # Sturges Rule
    k = int(1 + 3.322 * np.log(len(fit)))
    
    ax.hist(fit, bins=k, linewidth=0.5, edgecolor="black", color=data["color"])

    ax.set_title(data["title"])
    ax.set_xlabel(data["label"])
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.6)

def draw_ply(fig, data):
    """Plotly implementation"""
    import plotly.graph_objects as go
    fit = data["fitness"]
    fig: go.Figure
    fig.add_trace(go.Histogram(
        x=fit,
        marker=dict(
            color=data["color"],
                    line=dict(
                        color="#000000",
                        width=1)),

    ))
    fig.update_layout(yaxis_title=data["label"])
    fig.update_layout(title=data["title"])