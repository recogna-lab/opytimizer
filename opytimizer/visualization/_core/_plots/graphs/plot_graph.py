"""Graph and Tree visualization module.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import opytimizer.utils.exception as e
from opytimizer.visualization._core import fields as F


def _to_networkx(target) -> Tuple[nx.DiGraph, bool]:
    G = nx.DiGraph()

    if hasattr(target, "root") and target.root is not None:

        def _build_tree(node):
            node_id = id(node)
            G.add_node(node_id, label=str(node.name))
            for child in node.children:
                child_id = id(child)
                G.add_edge(node_id, child_id)
                _build_tree(child)

        _build_tree(target.root)
        return G, True

    if hasattr(target, "nodes") and hasattr(target, "edges"):
        for node in target.nodes:
            G.add_node(id(node), label=str(getattr(node, "name", node)))

        for edge in target.edges:
            G.add_edge(id(edge.source), id(edge.target))
        return G, False

    return G, False


def extract_data(
    result=None,
    fields: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict:
    if result is None:
        raise e.ValueError("No graph/tree provided.")

    target = getattr(result, "position", result)
    G, is_tree = _to_networkx(target)

    if is_tree:
        try:
            pos = graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G)
    else:
        pos = nx.circular_layout(G) if len(G) <= 20 else nx.spring_layout(G)

    fmap = F.resolve(fields)
    out: Dict = {}

    if F.wants(fmap, "nodes"):
        out["nodes"] = list(G.nodes())
    if F.wants(fmap, "edges"):
        out["edges"] = list(G.edges())
    if F.wants(fmap, "labels"):
        out["labels"] = nx.get_node_attributes(G, "label")
    if F.wants(fmap, "pos"):
        out["pos"] = pos
    if F.wants(fmap, "title"):
        default_title = "Tree Representation" if is_tree else "Graph Representation"
        out["title"] = kwargs.get("title") or default_title

    return out


def draw_mpl(ax, data: Dict) -> None:
    G = nx.DiGraph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])

    node_size = 1200

    nx.draw_networkx_nodes(
        G,
        data["pos"],
        ax=ax,
        node_size=node_size,
        node_color="#1f77b4",
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        G,
        data["pos"],
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=22,
        edge_color="#d62728",
        width=2.0,
        node_size=node_size,
        min_source_margin=22,
        min_target_margin=22,
        connectionstyle="arc3,rad=0.15",
    )
    nx.draw_networkx_labels(
        G,
        data["pos"],
        labels=data["labels"],
        ax=ax,
        font_size=9,
        font_color="white",
        font_weight="bold",
    )

    ax.margins(0.2)
    ax.set_title(data["title"])
    ax.axis("off")


def draw_ply(fig, data: Dict) -> None:
    import plotly.graph_objects as go

    for edge in data["edges"]:
        if edge[0] in data["pos"] and edge[1] in data["pos"]:
            x0, y0 = data["pos"][edge[0]]
            x1, y1 = data["pos"][edge[1]]

            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="#d62728",
                standoff=18,
                startstandoff=18,
            )

    node_x, node_y, texts = [], [], []
    for node in data["nodes"]:
        if node in data["pos"]:
            x, y = data["pos"][node]
            node_x.append(x)
            node_y.append(y)
            texts.append(data["labels"].get(node, ""))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=texts,
        textposition="middle center",
        hoverinfo="text",
        marker=dict(size=25, color="#1f77b4"),
        textfont=dict(color="white", size=10, family="Arial Black"),
    )

    fig.add_trace(node_trace)
    fig.update_layout(
        title=data["title"],
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
