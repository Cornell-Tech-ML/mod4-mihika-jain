"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiple 'x' by 'y'"""
    return x * y


def id(x: float) -> float:
    """Return the value of 'x'"""
    return x


def add(x: float, y: float) -> float:
    """Add 'x' to 'y'"""
    return x + y


def neg(x: float) -> float:
    """Return negated value of 'x'"""
    return -x


def lt(x: float, y: float) -> float:
    """Return 'x'<'y'"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if 'x' is equal to 'y'"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Calculate the max of 'x' and 'y'"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x and y are close within a tolerance of 1e-2."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function for 'x'"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function for 'x'"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Applies the natural logarithm to 'x'"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential function of 'x'"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal of 'x'"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """If f(x) = log computer d times f'(x)"""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """If f(x) = 1/x compute d times f'(x)"""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """If f(x) = relu compute d time f'(x)"""
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher order map

    Arguments:
    ---------
        fn: A callable function that takes a single argument and returns a value.
        iterable: An iterable (e.g., list, tuple) whose elements are passed to `func`.

    Returns:
    -------
        A list of results where each element is the result of applying `func` to the corresponding element in `iterable`.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combine elements from two iterables using a given function.

    Arguments:
    ---------
        fn: A callable function that takes two arguments and returns a value.
        iterable1: The first iterable whose elements are paired with elements from `iterable2`.
        iterable2: The second iterable whose elements are paired with elements from `iterable1`.

    Returns:
    -------
        A list of results where each element is the result of applying `func` to the corresponding elements from `iterable1` and `iterable2`.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value by applying a function cumulatively.

    Arguments:
    ---------
        fn: A callable function that takes two arguments and returns a value.
        iterable: An iterable whose elements are reduced by `func`.
        start: initial default val

    Returns:
    -------
    - A single value that is the result of applying `func` cumulatively to the elements of `iterable`, starting with `initial` if provided.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list."""
    return reduce(mul, 1.0)(ls)
