from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, plot_fn: Callable, data: Dict) -> Any: ...
    
    @abstractmethod
    def update_title(self, fig: Any, text: str) -> Any: ...
    
    @abstractmethod
    def save(self, fig: Any, path: str, **kwargs) -> None: ...
    
    @abstractmethod
    def show(self, fig: Any) -> None: ...
    
    @abstractmethod
    def to_html(self, fig: Any, **kwargs) -> str: ...
    
    
class MatplotlibRenderer(BaseRenderer):
    def render(self, plot_fn: Callable, data: Dict) -> Any:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = data.get("figsize", (8, 5)))
        plot_fn(ax, data)
        fig.tight_layout()
        return fig
    
    
    def update_title(self, fig: Any, text: str) -> Any:
        fig.axes[0].set_title(text)
        return fig
    
    def save(self, fig: Any, path: str, dpi: int = 300, **kwargs) -> None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        
    def show(self, fig: Any) -> None:
        import matplotlib.pyplot as plt
        plt.show()
        
    def to_html(self, fig: Any, dpi: int = 300, **kwargs) -> str:
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{b64}" />'
    
    

class PlotlyRenderer(BaseRenderer):
    def render(self, plot_fn: Callable, data: dict) -> Any:
        import plotly.graph_objects as go
        fig = go.Figure()
        plot_fn(fig, data)
        fig.update_layout(template="plotly_white")
        return fig

    def update_title(self, fig: Any, text: str) -> Any:
        fig.update_layout(title=text)
        return fig

    def save(self, fig: Any, path: str) -> None:
        if path.endswith(".html"): fig.write_html(path)
        else: fig.write_image(path)

    def show(self, fig: Any) -> None:
        fig.show()

    def to_html(self, fig: Any) -> str:
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    
_RENDERER_REGISTRY = {
    "matplotlib": MatplotlibRenderer, "mpl": MatplotlibRenderer,
    "plotly": PlotlyRenderer, "ply": PlotlyRenderer,
}