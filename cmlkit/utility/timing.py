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
    """Time a function multiple times and return results and statistics.

    Expects a function without arguments.
    """

    results = []

    times = np.zeros(repeats, dtype=float)
    for i in range(repeats):
        start = time.monotonic()
        res = f()
        results.append(res)
        end = time.monotonic()

        times[i] = end - start

    return (
        results,
        {
            "times": times.tolist(),
            "mean": np.mean(times).item(),
            "min": np.min(times).item(),
            "max": np.max(times).item(),
        },
    )
