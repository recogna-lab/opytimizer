from opytimizer.core.graph.primitive import Ephemeral, Primitive, Terminal


def test_primitive_arity():
    primitive = Primitive("add", lambda a, b: a + b, (int, int), int)

    assert primitive.arity == 2


def test_terminal_fields():
    terminal = Terminal("one", 1, int)

    assert terminal.name == "one"
    assert terminal.value == 1
    assert terminal.output_type is int


def test_ephemeral_fields():
    generator = lambda: 3
    ephemeral = Ephemeral("rand", generator, int)

    assert ephemeral.name == "rand"
    assert ephemeral.generator is generator
    assert ephemeral.output_type is int
