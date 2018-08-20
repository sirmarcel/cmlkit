import numpy as np


def log2(start, stop, n):
    """Generate a 1-d grid in base 2 logspace."""

    return np.logspace(start, stop, num=n, base=2.0)


def medium():
    """Log grid from 2^-18 to 2^20 with 39 steps."""
    return np.logspace(-18, 20, num=39, base=2.0)


def coarse():
    """Log grid from 2^-18 to 2^20 with 16 steps."""
    return np.logspace(-18, 20, num=16, base=2.0)
