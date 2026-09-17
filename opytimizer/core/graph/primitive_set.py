"""Typed registry of primitives/terminals.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import opytimizer.utils.exception as e
from opytimizer.core.graph.primitive import Ephemeral, Primitive, Terminal


class PrimitiveSet:
    """Registry of typed primitives, terminals and ephemeral constants.

    Args:
        name: Identifier for this set (useful when several problems/typed
            grammars coexist).
        root_type: The type that a tree generated from this set must
            produce at its root.

    """

    def __init__(self, name: str, root_type: Type) -> None:
        self.name = name
        self.root_type = root_type

        self._primitives: Dict[Type, List[Primitive]] = defaultdict(list)
        self._terminals: Dict[Type, List[Union[Terminal, Ephemeral]]] = defaultdict(list)

    def add_primitive(
        self,
        function: Callable[..., Any],
        input_types: Tuple[Type, ...],
        output_type: Type,
        name: Optional[str] = None,
    ) -> None:
        """Registers a typed function node.

        Args:
            function: Callable to be applied to the node's children.
            input_types: Expected type of each child, in order.
            output_type: Type produced by `function`.
            name: Optional display name (defaults to `function.__name__`).

        """

        self._primitives[output_type].append(
            Primitive(name or function.__name__, function, tuple(input_types), output_type)
        )

    def add_terminal(self, value: Any, output_type: Type, name: Optional[str] = None) -> None:
        """Registers a fixed leaf value."""

        self._terminals[output_type].append(Terminal(name or repr(value), value, output_type))

    def add_ephemeral_constant(
        self, name: str, generator: Callable[[], Any], output_type: Type
    ) -> None:
        """Registers a leaf that is freshly sampled from `generator` every
        time it's drawn during tree generation."""

        self._terminals[output_type].append(Ephemeral(name, generator, output_type))

    def primitives_of(self, output_type: Type) -> List[Primitive]:
        """Returns every primitive whose `output_type` matches."""

        return self._primitives.get(output_type, [])

    def terminals_of(self, output_type: Type) -> List[Union[Terminal, Ephemeral]]:
        """Returns every terminal/ephemeral whose `output_type` matches."""

        return self._terminals.get(output_type, [])

    def has_terminal(self, output_type: Type) -> bool:
        return bool(self.terminals_of(output_type))

    def validate(self) -> None:
        """Walks every registered primitive's `input_types` and raises if
        any of them has no matching terminal/primitive registered — i.e.,
        the grammar would eventually get stuck trying to close a branch.

        Call this once, right after registering everything.

        """

        known_types = set(self._primitives) | set(self._terminals)
        for primitives in self._primitives.values():
            for primitive in primitives:
                for input_type in primitive.input_types:
                    if input_type not in known_types:
                        raise e.ValueError(
                            f"Primitive `{primitive.name}` expects an input of type "
                            f"`{input_type}`, but no primitive/terminal in `{self.name}` "
                            "produces that type."
                        )
        if self.root_type not in known_types:
            raise e.ValueError(
                f"`root_type` `{self.root_type}` has no matching primitive/terminal "
                f"registered in `{self.name}`."
            )
