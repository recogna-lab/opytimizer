from typing import Callable, Dict
from opytimizer.visualization._core.result import PlotResult
from opytimizer.visualization._core.renderers import MatplotlibRenderer, BaseRenderer, _RENDERER_REGISTRY
class PlotRouter:
    default_backend: str = "matplotlib"
    
    def __init__(self, default_backend: str | None = None):
        if default_backend: self.default_backend = default_backend
        
    def dispatch(self, plot_fn_mpl: Callable, plot_fn_ply: Callable, data: Dict, backend: str | None = None, auto_show: bool = True, **kwargs) -> PlotResult:
        renderer = self._resolve_backend(backend or self.default_backend)
        plot_fn = plot_fn_mpl if isinstance(renderer, MatplotlibRenderer) else plot_fn_ply
        data = {**data, **kwargs}
        fig = renderer.render(plot_fn, data)
        return PlotResult(fig, renderer, auto_show=auto_show)
    
    
    def _resolve_backend(self, backend: str | type) -> BaseRenderer:
        if isinstance(backend, BaseRenderer): return backend
        if isinstance(backend, type) and issubclass(backend, BaseRenderer): return backend()
        if isinstance(backend, str):
            cls = _RENDERER_REGISTRY.get(backend.lower())
            if not cls: raise ValueError(f"Unknown backend: {backend}")
            return cls()
        raise TypeError("Backend must be str or BaseRenderer")

_router = PlotRouter()