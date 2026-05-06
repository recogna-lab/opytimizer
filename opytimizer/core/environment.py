"""
CPU/GPU core logic
""" 

from __future__ import annotations

import os
from enum import Enum
from typing_extensions import Literal

class Backend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"

class Environment:
    """Singleton thread-safe"""

    def _init_defaults(self) -> None:
        """Standard values initialization"""
        env_backend = os.getenv("XPYTHON_BACKEND", "cpu").lower()
        self._backend: Backend = (Backend.CUDA if env_backend == "cuda" else Backend.CPU)
        self._dtype: str = os.getenv("XPYTHON_DTYPE", "float64")


    # Public API

    @property
    def xp(self):
        """Returns current state-based proxy (np or cp)"""
        from opytimizer.core._backend import ArrayBackendProxy
        return ArrayBackendProxy(self)
    
    @property
    def backend(self) -> Backend:
        return self._backend
    
    @property
    def dtype(self) -> str:
        return self._dtype
    
    @property
    def use_cuda(self) -> bool:
        return self._backend is Backend.CUDA
    

    def set_backend(self, backend: Literal["cpu", "cuda"] | Backend) -> Environment:
        """Defines computational backend"""


        self._backend = Backend(str(backend).lower())
        return self
    
    def set_dtype(self, dtype: str) -> Environment:
        """Defines arrays data type"""
        self._dtype = dtype
        return self
    
    
    def reset(self) -> Environment:
        """Resets to the standard values"""
        self._init_defaults()
        return self
    

    # Internal
        
    def __repr__(self) -> str:
        return(
            f"Environment(backend={self._backend.value!r}, "
            f"dtype={self._dtype!r} "
        )
    
