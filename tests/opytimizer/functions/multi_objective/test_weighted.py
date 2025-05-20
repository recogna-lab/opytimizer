from opytimizer.functions.multi_objective import weighted


def test_weighted_weights():
    def square(x):
        return x**2

    new_weighted = weighted.WeightedFunction([square], [1.0])

    assert type(new_weighted.weights) == list


def test_weighted_weights_setter():
    def square(x):
        return x**2

    new_weighted = weighted.WeightedFunction([square], [1.0])

    try:
        new_weighted.weights = None
    except:
        new_weighted.weights = [1.0]

    try:
        new_weighted.weights = [1.0, 2.0]  # Tenta adicionar mais pesos que funções
    except:
        new_weighted.weights = [1.0]

    assert len(new_weighted.weights) == 1


def test_weighted_call():
    def square(x):
        return x**2

    assert square(2) == 4

    def cube(x):
        return x**3

    assert cube(2) == 8

    new_weighted = weighted.WeightedFunction(
        functions=[square, cube], weights=[0.5, 0.5]
    )

    assert new_weighted(2) == 6