"""Parametrisation schemes for symmetry functions.

A scheme is expected to return a list of symmetry function configs,
and a count of the radial and angular symmetry functions generated.

"""

from cmlkit.engine import parse_config


def rad_centered(n, cutoff):
    """Generate set of radial symmetry functions with broadenings on a grid.

    This parametrisation scheme is adapted from
    Gastegger et. al, J. Chem. Phys. 148, 241709 (2018).

    Args:
        n: number of symmetry functions to generate
        cutoff: cutoff radius

    """
    r0 = 0.5
    rn = cutoff - 1.0
    delta = (rn - r0) / (float(n - 1))

    sfs = [
        {"rad": {"cutoff": cutoff, "eta": 0.5 / (r0 + i * delta) ** 2, "mu": 0.0}}
        for i in range(n)
    ]

    return sfs, n, 0


def rad_shifted(n, cutoff):
    """Generate set of radial symmetry functions with means on a grid.

    This parametrisation scheme is adapted from
    Gastegger et. al, J. Chem. Phys. 148, 241709 (2018).

    Args:
        n: number of symmetry functions to generate
        cutoff: cutoff radius

    """

    r0 = 0.5
    rn = cutoff - 1.0
    delta = (rn - r0) / (float(n - 1))

    sfs = [
        {"rad": {"cutoff": cutoff, "eta": 0.5 / (delta) ** 2, "mu": r0 + i * delta}}
        for i in range(n)
    ]

    return sfs, n, 0


schemes = {rad_shifted.__name__: rad_shifted, rad_centered.__name__: rad_centered}
