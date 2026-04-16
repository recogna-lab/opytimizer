from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import opytimizer.utils.exception as e

def extract_data(result=None, *args: List[float], **kwargs) -> Dict:
    """
    Extracts SO convergence data. 
    If 'iterations' is provided (List[int]), it filters those specific steps.
    """
    if not args:
        raise e.ValueError("No convergence data provided.")

    requested_indices = kwargs.get("iterations")
    
    processed_curves = []
    for opt in args:
        if requested_indices is not None:
            # Filters specific points: e.g., [1, 10, 20...]
            processed_curves.append([opt[i - 1][1] for i in requested_indices if i < len(opt)])
            x_axis = requested_indices
        else:
            # Default: all iterations
            processed_curves.append([opt[i][1] for i in range(len(opt))])
            x_axis = list(range(1, len(opt) + 1))

    return {
        "curves": processed_curves,
        "x_axis": x_axis,
        "labels": kwargs.get("labels") or [f"Alg {i+1}" for i in range(len(args))],
        "title": kwargs.get("title") or "Convergence Analysis",
        "xlabel": kwargs.get("xlabel") or "Iteration",
        "ylabel": kwargs.get("ylabel") or "Best Fitness"
    }

def draw_mpl(ax, data: Dict):
    for curve, label in zip(data["curves"], data["labels"]):
        # Use x_axis to ensure points align correctly if indices are non-consecutive
        ax.plot(data["x_axis"], curve, label=label, marker='o' if len(curve) < 20 else None)
    
    ax.set_title(data["title"])
    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.legend()
    

def draw_ply(fig, data: Dict):
    """Plotly implementation for SO convergence."""
    import plotly.graph_objects as go
    
    for curve, label in zip(data["curves"], data["labels"]):
        fig.add_trace(go.Scatter(
            y=curve, 
            mode="lines", 
            name=label
        ))

    fig.update_layout(
        title=data["title"],
        xaxis_title=data["xlabel"],
        yaxis_title=data["ylabel"]
    )