"""Typed building blocks for strongly-typed genetic programming (STGP):
`Primitive` (function nodes), `Terminal` (fixed leaves) and `Ephemeral`
(dynamically generated leaves).
"""

from dataclasses import dataclass
from typing import Any, Callable, Tuple, Type


@dataclass
class Primitive:
    """An internal (function) node: applies `function` over `arity` typed
    children and produces a value of `output_type`.

    Args:
        name: Display name.
        function: Callable applied to the evaluated children.
        input_types: Expected type of each child, in order.
        output_type: Type produced by `function`.

    """

    name: str
    function: Callable[..., Any]
    input_types: Tuple[Type, ...]
    output_type: Type

    @property
    def arity(self) -> int:
        return len(self.input_types)


@dataclass
class Terminal:
    """A fixed leaf value.

    Args:
        name: Display name.
        value: The terminal's constant value.
        output_type: Type of `value`.

    """

    name: str
    value: Any
    output_type: Type


@dataclass
class Ephemeral:
    """A leaf whose value is (re)sampled from `generator` every time it is
    drawn during tree generation — e.g., a random constant.

    Args:
        name: Display name.
        generator: Zero-argument callable producing a fresh value.
        output_type: Type of the values `generator` produces.

    """

    name: str
    generator: Callable[[], Any]
    output_type: Type
