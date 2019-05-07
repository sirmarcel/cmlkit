"""Basic timing tools."""

from functools import wraps
import time
import numpy as np


def timed(f):
    """Wraps a function so it returns results and run time."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = f(*args, **kwargs)
        end = time.monotonic()
        return result, end - start

    return wrapper


def time_repeat(f, repeats=3):
    """Time a function multiple times and return statistics.

    Expects a function without arguments.
    """

    times = np.zeros(repeats, dtype=float)
    for i in range(repeats):
        start = time.monotonic()
        f()
        end = time.monotonic()

        times[i] = end - start

    return times, np.mean(times), np.min(times), np.max(times)
