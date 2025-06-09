"""Multi-objective visualization plots.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

import opytimizer.utils.exception as e


def plot_pareto_front(
    pareto_front: List,
    title: str = "Pareto Front",
    subtitle: str = "",
    xlabel: str = "f1",
    ylabel: str = "f2",
    zlabel: str = "f3",
    grid: bool = True,
    scatter: bool = True,
    line: bool = True,
    all_solutions: Optional[List] = None,
) -> None:
    """Plots the Pareto front, optionally showing all non-dominated solutions.

    Args:
        pareto_front: List of agents in the Pareto front.
        all_solutions: (optional) List of all agents in the population.
        title: Title of the plot.
        subtitle: Subtitle of the plot.
        xlabel: Axis `x` label.
        ylabel: Axis `y` label.
        zlabel: Axis `z` label.
        grid: If grid should be used or not.
        scatter: If scatter plot should be used or not.
        line: If line plot should be used or not.

    """

    n_dimensions = pareto_front[0].fit.shape[0]
    if n_dimensions == 3:
        # 3 dimensions
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    elif n_dimensions == 2:
        # 2 dimensions
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        # PF can't be shown
        raise e.ValueError("Only two and three-objetive problems can be ploted")

    if all_solutions is not None:
        f1_all = [agent.fit[0][0] for agent in all_solutions]
        f2_all = [agent.fit[1][0] for agent in all_solutions]

        if n_dimensions == 3:
            f3_all = [agent.fit[2][0] for agent in all_solutions]
            ax.scatter(
                f1_all, f2_all, f3_all, c="lightgray", alpha=0.5, label="All solutions"
            )
        else:
            ax.scatter(f1_all, f2_all, c="lightgray", alpha=0.5, label="All Solutions")

    f1 = [agent.fit[0][0] for agent in pareto_front]
    f2 = [agent.fit[1][0] for agent in pareto_front]

    if n_dimensions == 3:
        f3 = [agent.fit[2][0] for agent in pareto_front]

    idx = np.argsort(f1)
    f1 = np.array(f1)[idx]
    f2 = np.array(f2)[idx]
    if n_dimensions == 3:
        f3 = np.array(f3)[idx]

    ax.set(xlabel=xlabel, ylabel=ylabel)
    if n_dimensions == 3:
        ax.set(zlabel=zlabel)

    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")

    if grid:
        ax.grid()

    if scatter:
        if n_dimensions == 3:
            ax.scatter(f1, f2, f3, c="blue", alpha=0.8, label="Pareto Front")
        else:
            ax.scatter(f1, f2, c="blue", alpha=0.8, label="Pareto Front")

    if line:
        if n_dimensions != 3:
            ax.plot(f1, f2, "r--", alpha=0.4, label="Pareto Line")

    ax.legend()
    plt.show()


def plot_pareto_evolution(
    pareto_fronts: List[List],
    iterations: List[int],
    title: str = "Pareto Front Evolution",
    subtitle: str = "",
    xlabel: str = "f1",
    ylabel: str = "f2",
    grid: bool = True,
) -> None:
    """Plots the evolution of Pareto front over iterations.

    Args:
        pareto_fronts: List of Pareto fronts at different iterations.
        iterations: List of iteration numbers.
        title: Title of the plot.
        subtitle: Subtitle of the plot.
        xlabel: Axis `x` label.
        ylabel: Axis `y` label.
        grid: If grid should be used or not.

    """

    _, ax = plt.subplots(figsize=(7, 5))

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")

    if grid:
        ax.grid()

    colors = plt.cm.viridis(np.linspace(0, 1, len(iterations)))

    for i, (front, it) in enumerate(zip(pareto_fronts, iterations)):
        f1 = [agent.fit[0][0] for agent in front]
        f2 = [agent.fit[1][0] for agent in front]

        idx = np.argsort(f1)
        f1 = np.array(f1)[idx]
        f2 = np.array(f2)[idx]

        ax.plot(f1, f2, "--", alpha=0.4, color=colors[i], label=f"Iteration {it}")

    ax.legend()
    plt.show()
