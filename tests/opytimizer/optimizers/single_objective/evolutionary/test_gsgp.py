import random

from opytimizer.core.graph.primitive_set import PrimitiveSet
from opytimizer.optimizers.single_objective.evolutionary import gsgp
from opytimizer.spaces import tree


def _make_pset() -> PrimitiveSet:
    pset = PrimitiveSet("gsgp_test", root_type=float)
    pset.add_primitive(lambda a, b: a + b, (float, float), float, name="add")
    pset.add_primitive(lambda a, b: a * b, (float, float), float, name="mul")
    pset.add_terminal(1.0, float, name="one")
    pset.add_ephemeral_constant("rand", lambda: random.uniform(-1.0, 1.0), float)
    return pset


def _make_space(n_agents: int = 10) -> tree._SingleObjectiveTreeSpace:
    return tree.TreeSpace(
        n_agents=n_agents,
        n_objectives=1,
        pset=_make_pset(),
        min_depth=2,
        max_depth=4,
    )


def _evaluate(space) -> None:
    for agent in space.agents:
        agent.fit = agent.position.evaluate()


def test_gsgp_mutate():
    new_gsgp = gsgp.GSGP()
    space = _make_space()

    original = space.agents[0].position
    mutated = new_gsgp._mutate(original)

    assert mutated is not original
    assert mutated.root is not None

    assert any(node.name == "SUM" for node in mutated.nodes)
    assert mutated.evaluate() is not None

    assert original.evaluate() is not None


def test_gsgp_mutation():
    new_gsgp = gsgp.GSGP()
    space = _make_space()
    _evaluate(space)

    new_gsgp._mutation(space)

    for agent in space.agents:
        agent.position.evaluate()


def test_gsgp_cross():
    new_gsgp = gsgp.GSGP()
    space = _make_space()

    offspring = new_gsgp._cross(space.agents[0].position, space.agents[1].position)

    assert offspring is not None
    assert offspring.root is not None
    assert any(node.name == "SUM" for node in offspring.nodes)
    assert isinstance(offspring.evaluate(), float)


def test_gsgp_crossover():
    new_gsgp = gsgp.GSGP()
    space = _make_space()
    _evaluate(space)

    new_gsgp._crossover(space)

    for agent in space.agents:
        agent.position.evaluate()


def test_gsgp_inherits_gp_reproduction():
    new_gsgp = gsgp.GSGP()
    space = _make_space()
    _evaluate(space)

    new_gsgp._reproduction(space)

    for agent in space.agents:
        agent.position.evaluate()
