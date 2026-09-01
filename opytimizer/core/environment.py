"""
CPU/GPU core logic
""" 

from __future__ import annotations

from enum import Enum
from typing_extensions import Literal
from opytimizer.core._backend import ArrayBackendProxy
from opytimizer.utils import logging

logger = logging.get_logger(__name__)
class Backend(str, Enum):
    NUMPY = "numpy"
    CUPY = "cupy"

class Environment:
    """Wrapper API class for backend control logic"""

    def __init__(self, device: Literal['numpy', 'cupy'] = 'numpy', dtype: str = 'float32'):

        
        self.set_backend(device) 
        self._dtype: str = dtype

    def _init_defaults(self) -> None:
        """Standard values initialization"""
        
        self._backend: Backend = Backend.NUMPY
        self._dtype: str = 'float32'


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
        return self._backend is Backend.CUPY
    

    def set_backend(self, backend: Literal["numpy", "cupy"] | Backend) -> Environment:
        """Defines computational backend"""
        self._backend = Backend(str(backend).lower())
        
        try:
            _ = ArrayBackendProxy(self)._get_module()
        except ImportError:
            logger.info("The current device does not have CuPy installed. Backend switched to numpy.")
            self._backend = Backend('numpy')
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
    
