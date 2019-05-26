"""Prepare RuNNer input files."""


import numpy as np
import time

from cmlkit import get_scratch
from cmlkit.engine import compute_hash, parse_config
from cmlkit.utility import charges_to_elements


def prepare_task(data, config):
    tid = compute_hash(time.time() + np.random.rand())

    folder = get_scratch() / f"runner_{tid}"
    folder.mkdir(parents=True)  # will raise error if already exists!

    with open(folder / "input.data", "w+") as f:
        datafile = make_datafile(data)
        f.write(datafile)

    with open(folder / "input.nn", "w+") as f:
        infile = make_infile(config)
        f.write(infile)

    return folder


def make_infile(config):
    lines = []

    # boilerplate
    lines.append("nn_type_short 1")
    lines.append("runner_mode 1")
    lines.append("parallel_mode 0")
    lines.append("energy_threshold 100.0")
    lines.append("bond_threshold 0.0")
    lines.append("use_short_nn")
    lines.append("global_hidden_layers_short 1")
    lines.append("global_nodes_short 1")
    lines.append("global_activation_short t")
    lines.append("use_atom_charges")
    lines.append("test_fraction 0.0")
    lines.append("cutoff_type 1")

    symbol_elements = [charges_to_elements[elem] for elem in config["elems"]]
    lines.append(f"number_of_elements {len(symbol_elements)}")
    lines.append(f"elements {' '.join(symbol_elements)}")
    for elem in symbol_elements:
        lines.append(f"atom_energy {elem} 0 ")

    for symmf in config["universal"]:
        lines.append(
            "global_symfunction_short " + " ".join(map(str, make_universal_sf(symmf)))
        )

    for symmf in config["elemental"]:
        lines.append("symfunction_short " + " ".join(map(str, make_elemental_sf(symmf))))

    return "\n".join(lines)


def make_elemental_sf(config):
    """Generate a symmetry function applied to specific elements."""

    kind, inner = parse_config(config)

    if kind == "rad":
        return _make_elemental_rad(**inner)

    if kind == "ang":
        return _make_elemental_ang(**inner)

    else:
        raise ValueError(
            f"Elemental symmetry function kind {kind} is not yet implemented."
        )


def _make_elemental_rad(cutoff, eta, mu, z1, z2):
    return [z1, 2, z2, eta, mu, cutoff]


def _make_elemental_ang(cutoff, eta, zeta, lambd, z1, z2, z3):
    return [z1, 3, z2, z3, eta, lambd, zeta, cutoff]


def make_universal_sf(config):
    """Generate a symmetry function applied to all elements."""

    kind, inner = parse_config(config)

    if kind == "rad":
        return _make_universal_rad(**inner)

    if kind == "ang":
        return _make_universal_ang(**inner)

    else:
        raise ValueError(
            f"universal symmetry function kind {kind} is not yet implemented."
        )


def _make_universal_rad(cutoff, eta, mu):
    return [2, eta, mu, cutoff]


def _make_universal_ang(cutoff, eta, zeta, lambd):
    return [3, eta, lambd, zeta, cutoff]


def make_datafile(data):
    lines = []

    for i in range(data.n):
        z = data.z[i]
        r = data.r[i]

        lines.append("begin")
        if data.b is not None:
            for v in data.b[i]:
                lines.append(f"lattice {v[0]} {v[1]} {v[2]}")

        for j in range(len(z)):
            rs = r[j]
            zs = z[j]
            lines.append(
                f"atom {rs[0]} {rs[1]} {rs[2]} {charges_to_elements[zs]} 0.0 0.0 0.0 0.0 0.0"
            )

        lines.append("end")

    return "\n".join(lines)
