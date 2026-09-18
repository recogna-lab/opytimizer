import random

from opytimizer.core.graph.primitive import Ephemeral, Primitive, Terminal


def test_primitive_arity():
    def _add(a, b):
        return a + b

    primitive = Primitive("add", _add, (int, int), int)

    assert primitive.arity == 2


def test_terminal_fields():
    terminal = Terminal("one", 1, int)

    assert terminal.name == "one"
    assert terminal.value == 1
    assert terminal.output_type is int


def test_ephemeral_fields():
    def _int_generator():
        return random.randint(0, 3)

    ephemeral = Ephemeral("rand", _int_generator, int)

    assert ephemeral.name == "rand"
    assert ephemeral.generator is _int_generator
    assert ephemeral.output_type is int
