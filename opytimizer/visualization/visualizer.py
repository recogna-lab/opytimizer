from typing import List, FrozenSet, Optional, Set, Union
import importlib

from opytimizer.core import Agent
from opytimizer.visualization._core.router import PlotRouter, MatplotlibRenderer
from opytimizer.visualization._core.result import PlotResult

_router = PlotRouter(default_backend="matplotlib")

_MAPPING: dict[str, str] = {
    "agents": (
        "opytimizer.visualization._core._plots"
        ".multi_objective.plot_agents"
    ),
    "pareto_front_evolution": (
        "opytimizer.visualization._core._plots"
        ".multi_objective.plot_pareto_front_evolution"
    ),
    "pareto_front_comparison": (
        "opytimizer.visualization._core._plots"
        ".multi_objective.plot_pareto_front_comparison"
    ),
    "population_distribution_histogram": (
        "opytimizer.visualization._core._plots"
        ".plot_population_distribution_histogram"
    ),
    "convergence": (
        "opytimizer.visualization._core._plots"
        ".single_objective.plot_convergence"
    ),
}

FieldsParam = Optional[Union[Set[str], FrozenSet[str]]]

def _call_plot(name, result, backend=None, *args, **kwargs) -> PlotResult:
   
    resolved_backend = backend or _router.default_backend
    renderer = _router._resolve_backend(resolved_backend)

    draw_mpl, draw_ply, data = _load_plot(name, result, *args, **kwargs)

    target_fn = draw_mpl if isinstance(renderer, MatplotlibRenderer) else draw_ply
    return PlotResult(renderer.render(target_fn, data), renderer)


# Public API Functions

def plot_agents(
    result, 
    backend: str | None = None, 
    title: str = "Agents",
    color: str = "#2ca02c",
    labels: List[str] | None = None,
    **kwargs
) -> PlotResult:
    """Plots a 2D or 3D Agents from a list of agents.

    Args:
        result: List of agents.
        backend: Visualization backend ('matplotlib' or 'plotly').
        title: Plot title.
        color: Hex color string for the scatter points.
        labels: List of labels for the objective axes.
        **kwargs: Additional plotting parameters.
    """
    return _call_plot("agents", result, backend, title=title, color=color, labels=labels, **kwargs)


def plot_pareto_front_evolution(
    result: List[List[Agent]], 
    backend: str | None = None, 
    title: str = "Pareto Front Evolution",
    cmap: str = "viridis",
    labels: List[str] | None = None,
    iterations: List[int] | None = None,
    **kwargs
) -> PlotResult:
    """Plots the progression of the Pareto Front across iterations.

    Args:
        result: List of lists containing agents for each recorded iteration.
        backend: Visualization backend ('matplotlib' or 'plotly').
        title: Plot title.
        cmap: Matplotlib colormap name for temporal gradient.
        labels: List of labels for the objective axes.
        iterations: Specific iteration indices to be plotted.
        **kwargs: Additional plotting parameters.
    """
    return _call_plot("pareto_front_evolution", result, backend, title=title, cmap=cmap, labels=labels, iterations=iterations, **kwargs)

def plot_pareto_front_comparison(
    *args, 
    backend: str | None = None, 
    title: str = "Pareto Front Comparison",
    labels: List[str] | None = None,
    obj_labels: List[str] | None = None,
    **kwargs
) -> PlotResult:
    """Compares final Pareto Fronts from multiple Multi-Objective optimizers.

    Args:
        *args: Multiple lists of agents (one list per algorithm).
        backend: Visualization backend ('matplotlib' or 'plotly').
        title: Plot title.
        labels: Names of the algorithms for the legend.
        obj_labels: List of labels for the objective axes.
        **kwargs: Additional plotting parameters.
    """
    return _call_plot("pareto_front_comparison", None, backend, *args, title=title, labels=labels, obj_labels=obj_labels, **kwargs)


def plot_population_distribution_histogram(
    result, 
    backend: str | None = None, 
    target: int = 0,
    title: str = None,
    color: str = "#2ca02c",
    label: str | None = None,
    **kwargs
) -> PlotResult:
    """Plots a fitness-based histogram showing the population distribution.

    Args:
        result: Population or list of agents.
        backend: Visualization backend ('matplotlib' or 'plotly').
        target: Index of the objective or variable to distribute (1 ... N).
        title: Plot title.
        color: Hex color string for the histogram bars.
        label: Label for the data series.
        **kwargs: Additional plotting parameters.
    """
    return _call_plot("population_distribution_histogram", result, backend, target=target, title=title, color=color, label=label, **kwargs)
    
def plot_convergence(
    *args, 
    backend: str | None = None, 
    title: str | None = "Convergence Comparison",
    labels: List[str] | None = None,
    x_axis: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    iterations: List[int] | None = None,
    **kwargs
) -> PlotResult:
    """Plots fitness convergence comparison between Single-Objective optimizers.

    Args:
        *args: Multiple lists or arrays of fitness values over time.
        backend: Visualization backend ('matplotlib' or 'plotly').
        title: Plot title.
        labels: Names of the algorithms or variables for the legend.
        x_axis: Custom values for the x-axis.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        iterations: Specific iteration indices to be plotted.
        **kwargs: Additional plotting parameters.
    """
    return _call_plot("convergence", None, backend, *args, title=title, labels=labels, x_axis=x_axis, xlabel=xlabel, ylabel=ylabel, iterations=iterations,**kwargs)
    
    
def _load_plot(name, result, *args, **kwargs):
    module = importlib.import_module(_MAPPING[name])
    data = module.extract_data(result, *args, **kwargs) 
    return module.draw_mpl, module.draw_ply, data