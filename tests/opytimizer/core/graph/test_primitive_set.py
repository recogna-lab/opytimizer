import pytest

from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.utils import exception as e


def test_add_and_retrieve_primitive():
    pset = PrimitiveSet("test", float)

    pset.add_primitive(lambda a: a, (float,), float, "identity")

    primitives = pset.primitives_of(float)

    assert len(primitives) == 1
    assert primitives[0].name == "identity"
    assert primitives[0].input_types == (float,)
    assert primitives[0].output_type is float


def test_add_and_retrieve_terminal():
    pset = PrimitiveSet("test", float)
    pset.add_terminal(1.5, float, "one")

    terminals = pset.terminals_of(float)

    assert len(terminals) == 1
    assert terminals[0].name == "one"
    assert terminals[0].value == 1.5
    assert pset.has_terminal(float)


def test_add_ephemeral_constant():
    pset = PrimitiveSet("test", float)
    pset.add_ephemeral_constant("rand", lambda: 7.0, float)

    ephemeral = pset.terminals_of(float)[0]

    assert ephemeral.name == "rand"
    assert ephemeral.generator() == 7.0


def test_unknown_type_returns_empty_list():
    pset = PrimitiveSet("test", float)

    assert pset.primitives_of(int) == []
    assert pset.terminals_of(int) == []
    assert not pset.has_terminal(int)


def test_validate_accepts_valid_grammar(primitive_set):
    primitive_set.validate()


def test_validate_rejects_missing_input_type():
    pset = PrimitiveSet("test", float)
    pset.add_primitive(lambda x: x, (int,), float, "needs_int")

    with pytest.raises(e.ValueError):
        pset.validate()


def test_validate_rejects_missing_root_type():
    pset = PrimitiveSet("test", float)
    pset.add_terminal(1, int, "one")

    with pytest.raises(e.ValueError):
        pset.validate()
