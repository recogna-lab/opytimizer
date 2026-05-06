"""
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from opytimizer.core.environment import Backend, Environment



class ArrayBackendProxy:
 
    __slots__ = ("_env",)
    
    def __init__(self, env: Environment):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name: str) -> Any:
        if name in ("_env", "_get_module"):
            return object.__getattribute__(self, name)
            
        mod = self._get_module()
        return getattr(mod, name)
    
    def _get_module(self) -> ModuleType:
        from opytimizer.core.environment import Backend
        env = object.__getattribute__(self, "_env")
        current = env.backend
        
        if current is Backend.CUDA:
            try:
                return importlib.import_module("cupy")
            except ImportError:
                return importlib.import_module("numpy")
        return importlib.import_module("numpy")
    
 
    def __dir__(self) -> list[str]:
        return dir(self._get_module())
 
    @property
    def __wrapped__(self) -> ModuleType:
        return self._get_module()
 
    def __repr__(self) -> str:
        env = object.__getattribute__(self, "_env")
        mod = self._get_module()
        return f"<ArrayBackendProxy(env={id(env)}) → {mod.__name__}>"
 
 