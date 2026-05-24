from __future__ import annotations

import numpy as np

from typing import Union, TYPE_CHECKING, List

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
    _HAS_CUPY: bool = True
except ImportError:
    cp = None
    _HAS_CUPY = False

ArrayLike = Union[np.ndarray, "cp.ndarray"]


def _is_gpu(arr: "cp.ndarray") -> np.ndarray:
    return _HAS_CUPY and isinstance(arr, cp.ndarray)


def _pinned_get(arr: "cp.ndarray") -> np.ndarray:
    pinned = cp.cuda.alloc_pinned_memory(arr.nbytes)
    out = np.frombuffer(pinned, dtype=arr.dtype, count=arr.size).reshape(arr.shape)
    arr.get(out=out)
    return out.copy()


def _async_get(arr: "cp.ndarray", stream: "cp.cuda.Stream") -> np.ndarray:
    pinned = cp.cuda.alloc_pinned_memory(arr.nbytes)
    out = np.frombuffer(pinned, dtype=arr.dtype, count=arr.size).reshape(arr.shape)
    arr.get(out=out, stream=stream)
    return out

def to_numpy(arr: ArrayLike) -> np.ndarray:
    if _is_gpu(arr):
        return _pinned_get(arr)
    return np.asarray(arr)

def agents_to_matrix(agents: List, *, async_transfer: bool = True) -> np.ndarray:
    if not agents:
        return np.empty((0, 0))
    
    sample_fit = agents[0].fit

    if _is_gpu(sample_fit):

        gpu_matrix = cp.stack([a.fit for a in agents])

        if async_transfer:
            stream = cp.cuda.Stream(non_blocking=True)
            raw = _async_get(gpu_matrix, stream)
            stream.synchronize()
            return raw.copy()
        else:
            return _pinned_get(gpu_matrix)
        

    fits = [
        a.fit.get() if hasattr(a.fit, "get") else np.asarray(a.fit)
        for a in agents
    ]

    return np.stack(fits)


def batch_agents_to_matrices(
        agent_groups: List[List],
        *,
        async_transfer: bool = True,
) -> List[np.ndarray]:
    
    if not agent_groups:
        return []
    
    sizes = [len(g) for g in agent_groups]
    flat_agents = [ag for group in agent_groups for ag in group]

    if not flat_agents:
        return [np.empty((0, 0)) for _ in agent_groups]
    
    sample_fit = flat_agents[0].fit

    if _is_gpu(sample_fit):
        gpu_all = cp.stack([a.fit for a in flat_agents])
        
        if async_transfer:
            stream = cp.cuda.Stream(non_blocking=True)
            raw = _async_get(gpu_all, stream)
            stream.synchronize()
            all_matrix = raw.copy()
        else:
            all_matrix = _pinned_get(gpu_all)

    else:
        fits = [
            a.fit.get() if hasattr(a.fit, "get") else np.asarray(a.fit)
            for a in flat_agents
        ]
        all_matrix = np.stack(fits)

    split_indices = np.cumsum(sizes)[:-1]
    return np.split(all_matrix, split_indices)


def batch_to_numpy(arrays: List[ArrayLike]) -> List[np.ndarray]:

    if not _HAS_CUPY:
        return [np.asarray(a) for a in arrays]


    gpu_indices = [i for i, a in enumerate(arrays) if _is_gpu(a)]
    if not gpu_indices:
        return [np.asarray(a) for a in arrays]
    
    stream = cp.cuda.Stream(non_blocking=True)
    pending: dict[int, np.ndarray] = {
        i: _async_get(arrays[i], stream) for i in gpu_indices
    }
    stream.synchronize()

    result: List = list(arrays)
    for i, raw in pending.items():
        result[i] = raw.copy()
    return result



