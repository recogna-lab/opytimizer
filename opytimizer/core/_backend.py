"""
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

from opytimizer.utils import logging

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from opytimizer.core.environment import Environment

# Process-wide cache: backend value ("numpy" / "cupy") -> imported module.
# `ArrayBackendProxy.__getattr__` runs on EVERY `xp.<attr>` access
_MODULE_CACHE: dict = {}


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
        env = object.__getattribute__(self, "_env")
        key = env.backend.value
        module = _MODULE_CACHE.get(key)
        if module is None:
            module = importlib.import_module("cupy" if key == "cupy" else "numpy")
            _MODULE_CACHE[key] = module

        return module

    def __dir__(self) -> list[str]:
        return dir(self._get_module())

    @property
    def __wrapped__(self) -> ModuleType:
        return self._get_module()

    def __repr__(self) -> str:
        env = object.__getattribute__(self, "_env")
        mod = self._get_module()
        return f"<ArrayBackendProxy(env={id(env)}) → {mod.__name__}>"
