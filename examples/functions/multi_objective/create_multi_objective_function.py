from opytimizer.core import Function


# Defines some test functions
def test_function1(z):
    return z + 2


def test_function2(z):
    return z + 5


# Declares `x`
x = 0

# Agora pode passar a lista de funções diretamente!
h = Function([test_function1, test_function2])

# Testing out your new Function class
print(f"x: {x}")
print(f"f(x): {h(x)[0]}")  # Primeiro objetivo
print(f"g(x): {h(x)[1]}")  # Segundo objetivo
print(f"h(x) = [f(x), g(x)]: {h(x)}")
