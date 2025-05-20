from opytimizer.optimizers.single_objective.misc import gs


def test_gs():
    new_gs = gs.GS()

    assert new_gs.built is True
