from __future__ import annotations
 
from typing import FrozenSet, Iterable, Optional

FieldMap = Optional[FrozenSet[str]]

ALL: FieldMap = None

def resolve(fields: Optional[Iterable[str]]) -> FieldMap:

    if fields is None:
        return None
    return frozenset(fields)


def wants(fmap: FieldMap, key: str) -> bool:
    return fmap is None or key in fmap

def wants_any(fmap: FieldMap, *keys: str) -> bool:
    return any(wants(fmap, k) for k in keys)

def wants_all(fmap: FieldMap, *keys: str) -> bool:
    return all(wants(fmap, k) for k in keys)

def subset(fmap: FieldMap, *keys: str) -> FieldMap:

    if fmap is None:
        return resolve(keys)
    return fmap & frozenset(keys)