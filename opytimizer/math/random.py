import numpy as np
from typing import Any, Optional, Union

def generate_binary_random_number(size: Union[int, tuple] = 1, xp: Any = np) -> np.ndarray:
    """Generates a binary random number or array based on an uniform distribution.

    Args:
        size: Size of array.
        xp: Array backend (numpy or cupy).

    Returns:
        (np.ndarray): A binary random number or array.
    """
    return xp.round(xp.random.uniform(0, 1, size))

def generate_exponential_random_number(scale: float = 1.0, size: Union[int, tuple] = 1, xp: Any = np) -> np.ndarray:
    """Generates a random number or array based on an exponential distribution.

    Args:
        scale: Scaling of the distribution.
        size: Size of array.
        xp: Array backend (numpy or cupy).

    Returns:
        (np.ndarray): An exponential random number or array.
    """
    return xp.random.exponential(scale, size)

def generate_gamma_random_number(
    shape: float = 1.0, scale: float = 1.0, size: Union[int, tuple] = 1, xp: Any = np
) -> np.ndarray:
    """Generates an Erlang distribution based on gamma values.

    Args:
        shape: Shape parameter.
        scale: Scaling of the distribution.
        size: Size of array.
        xp: Array backend (numpy or cupy).

    Returns:
        (np.ndarray): An Erlang distribution array.
    """
    return xp.random.gamma(shape, scale, size)

def generate_integer_random_number(
    low: int = 0,
    high: int = 1,
    exclude_value: Optional[int] = None,
    size: Union[int, tuple] = None,
    xp: Any = np
) -> np.ndarray:
    """Generates a random number or array based on an integer distribution.

    Args:
        low: Lower interval.
        high: Higher interval.
        exclude_value: Value to be excluded from array.
        size: Size of array.
        xp: Array backend (numpy or cupy).

    Returns:
        (np.ndarray): An integer random number or array.
    """
    integer_array = xp.asarray(xp.random.randint(low, high, size))

    if exclude_value is not None:
        mask = (integer_array == exclude_value)
        while xp.any(mask):
            fill_values = xp.random.randint(low, high, xp.sum(mask).item())
            integer_array[mask] = fill_values
            mask = (integer_array == exclude_value)

    return integer_array

def generate_uniform_random_number(
    low: Union[float, np.ndarray] = 0.0, 
    high: Union[float, np.ndarray] = 1.0, 
    size: Union[int, tuple] = 1, 
    xp: Any = np
) -> np.ndarray:
    """Generates a random number or array based on a uniform distribution.

    Args:
        low: Lower interval.
        high: Higher interval.
        size: Size of array.
        xp: Array backend (numpy or cupy).

    Returns:
        (np.ndarray): A uniform random number or array.
    """
    return xp.random.uniform(low, high, size)

def generate_gaussian_random_number(
    mean: float = 0.0,
    variance: float = 1.0,
    size: Union[int, tuple] = 1,
    xp: Any = np
) -> np.ndarray:
    """Generates a random number or array based on a gaussian distribution.

    Args:
        mean: Gaussian's mean value.
        variance: Gaussian's variance value.
        size: Size of array.
        xp: Array backend (numpy or cupy).

    Returns:
        (np.ndarray): A gaussian random number or array.
    """
    return xp.random.normal(mean, variance, size)