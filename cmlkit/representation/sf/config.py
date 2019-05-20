"""Config syntax for symmetry functions.

The RuNNer interface provided by `cmlkit` (see `runner.py`)
expects a config dictionary as input. This module helps with
preparing this config.

The config has the following keys:
    'universal': configs of universal SFs
    'elemental': configs of elemental SFs
    'dim': dimensionality of descriptor
    'elems': elements to be computed

Here, we
a) Normalize the individual SF configs, dropping `None` and making
    sure that only known configs are passed,
b) Apply parametrisation schemes,
c) Infer (if possible) the `dim` of the descriptor.

Note that we use "universal" because `global` is a keyword.

"""

from cmlkit.engine import parse_config
from .parametrization import schemes


def prepare_config(elems, elemental=[], universal=[], dim=None):
    """Prepare the config dictionary.

    Args:
        elems: List of elements (as charges).
        elemental: List of configs of elemental symmetry functions.
            Entries which are `None` are ignored.
        universal: List of configs of universal symmetry functions, or
            of parametrization schemes (see `parametrization` module).
            Entries which are `None` are ignored.
        dim: Dimensionality of descriptor.

    Returns:
        Config dictionary.

    """

    elemental = normalize_elemental_sfs(elemental)
    universal, count_rad, count_ang = normalize_universal_sfs(universal)

    if dim is None:
        assert (
            len(elemental) == 0
        ), "Cannot infer number of symmetry functions if elemental SFs are defined."
        dim = infer_dim(count_rad, count_ang, len(elems))

    return {"elemental": elemental, "universal": universal, "dim": dim, "elems": elems}


def normalize_universal_sfs(sfs):
    result = []
    count_rad = 0
    count_ang = 0

    for sf in sfs:
        if sf is not None:
            kind, inner = parse_config(sf)

            if kind == "rad":
                result.append(sf)
                count_rad += 1
            elif kind == "ang":
                result.append(sf)
                count_ang += 1
            elif kind in schemes:
                generated, tmp_rad, tmp_ang = schemes[kind](**inner)
                result.extend(generated)
                count_rad += tmp_rad
                count_ang += tmp_ang
            else:
                raise ValueError(
                    f"Don't know how to deal with universal symmetry function config {sf}."
                )

    return result, count_rad, count_ang


def normalize_elemental_sfs(sfs):
    result = []

    for sf in sfs:
        if sf is not None:
            kind, inner = parse_config(sf)

            if kind in ["rad", "ang"]:
                result.append(sf)
            else:
                raise ValueError(
                    f"Don't know how to deal with elemental symmetry function config {sf}."
                )

    return result


def infer_dim(n_twobody, n_threebody, n_elems):
    return int(n_twobody * n_elems + n_threebody * n_elems * (n_elems + 1) / 2)
