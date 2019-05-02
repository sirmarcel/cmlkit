"""Basic timing tools"""

from functools import wraps
import time


def timed(f):
    """Wraps a function so it returns results and run time"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = f(*args, **kwargs)
        end = time.monotonic()
        return result, end - start

    return wrapper
