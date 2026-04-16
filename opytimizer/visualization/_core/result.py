from __future__ import annotations

from typing import Any
from opytimizer.visualization._core.renderers import BaseRenderer

class PlotResult:
    def __init__(self, fig: Any, renderer: BaseRenderer, auto_show: bool = True):
        self._fig = fig
        self._renderer = renderer
        if auto_show: self._auto_display()
        
        
    @property
    def figure(self) -> Any: return self._fig
    
    def title(self, text: str) -> PlotResult:
        self._fig = self._renderer.update_title(self._fig, text)
        return self
    
    def show(self) ->PlotResult:
        return self._renderer.show(self._fig)
        return self
    
    def save(self, path: str, **kwargs) -> PlotResult:
        if 'dpi' in kwargs:
            self._renderer.save(self._fig, path, dpi=kwargs['dpi'])
        else:
            self._renderer.save(self._fig, path)
            
        return self
    
    def _auto_display(self) -> None:
        try:
            ip = __import__("IPython").get_ipython()
            if ip:
                from IPython.display import display
                display(self._fig)
        except (ImportError, AttributeError):pass
        
        