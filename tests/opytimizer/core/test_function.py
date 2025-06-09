import numpy as np

from opytimizer.core import function


def pointer(x):
    return x


assert pointer(1) == 1


def test_function_name():
    new_function = function.Function(pointer)

    assert new_function.name == "pointer"


def test_function_name_setter():
    new_function = function.Function(pointer)

    try:
        new_function.name = 1
    except:
        new_function.name = "pointer"

    assert new_function.name == "pointer"


def test_function_pointer():
    new_function = function.Function(pointer)

    assert new_function.pointer.__name__ == "pointer"


def test_function_pointer_setter():
    new_function = function.Function(pointer)

    try:
        new_function.pointer = "a"
    except:
        new_function.pointer = callable

    assert (
        new_function.pointer.__class__.__name__ == "builtin_function_or_method"
        or "builtin_function"
    )


def test_function_built():
    new_function = function.Function(pointer)

    assert new_function.built is True


def test_function_built_setter():
    new_function = function.Function(pointer)

    new_function.built = False

    assert new_function.built is False


def test_function_call():
    def square(x):
        return np.sum(x**2)

    assert square(2) == 4

    def square2(x, y):
        return x**2 + y**2

    assert square2(2, 2) == 8

    new_function = function.Function(square)

    assert new_function(np.zeros(2)) == 0

    try:
        new_function = function.Function(square2)
    except:
        new_function = function.Function(square)

    assert new_function.name == "square"


def test_function():
    class Square:
        def __call__(self, x):
            return np.sum(x**2)

    s = Square()

    assert s(2) == 4

    new_function = function.Function(s)

    assert new_function.name == "Square"


def test_function_multi_objective_name():
    def f1(x):
        return np.sum(x**2)

    def f2(x):
        return np.sum(x)

    funcs = [f1, f2]

    new_function = function.Function(funcs)

    assert new_function.name == "MultiObjectiveFunction"


def test_function_multi_objective_pointer_is_callable():
    def f1(x):
        return np.sum(x**2)

    def f2(x):
        return np.sum(x)

    funcs = [f1, f2]

    new_function = function.Function(funcs)

    assert callable(new_function.pointer)


def test_function_multi_objective_built():
    def f1(x):
        return np.sum(x**2)

    def f2(x):
        return np.sum(x)

    funcs = [f1, f2]

    new_function = function.Function(funcs)

    assert new_function.built is True


def test_function_multi_objective_call():
    def f1(x):
        return np.sum(x**2)

    def f2(x):
        return np.sum(x)

    funcs = [f1, f2]

    x = np.array([1, 2, 3])

    results = [f(x) for f in funcs]

    new_function = function.Function(funcs)

    np.testing.assert_array_equal(new_function(x), np.array(results))


def test_function_multi_objective_n_objectives():
    def f1(x):
        return np.sum(x**2)

    def f2(x):
        return np.sum(x)

    funcs = [f1, f2]

    new_function = function.Function(funcs)

    assert new_function.n_objectives == 2


def test_function_multi_objective_pointer_call():
    def f1(x):
        return np.sum(x**2)

    def f2(x):
        return np.sum(x)

    funcs = [f1, f2]

    x = np.array([1, 2, 3])

    results = [f(x) for f in funcs]

    new_function = function.Function(funcs)

    np.testing.assert_array_equal(new_function.pointer(x), np.array(results))
