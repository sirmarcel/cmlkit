"""Call RuNNer, collect results."""

import numpy as np
import subprocess

from cmlkit import runner_path


def run_task(folder, timeout=None):
    # this will raise an error if ruNNer does not
    # terminate with 0 or times out
    finished = subprocess.run(
        [runner_path, str(folder / "input.data"), str(folder / "input.nn")],
        cwd=folder,
        timeout=timeout,
        check=True,
        encoding="utf-8",
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    with open(folder / "STDOUT", "w+") as f:
        f.write(finished.stdout)
    with open(folder / "STDERR", "w+") as f:
        f.write(finished.stderr)

    if "ERROR" in finished.stdout:
        # TODO: make more useful errors
        raise Exception(
            f"ruNNer did not terminate correctly. Here is stdout:\n{finished.stdout}"
        )

    return finished.stdout, finished.stderr


def read_result(data, folder, config):
    elems = list(config["elems"])  # so we can use the index method later
    total_elems = len(elems)

    dim = config["dim"]

    total_atoms = data.info["total_atoms"]
    atoms_by_system = data.info["atoms_by_system"]

    descriptors = []

    i_system = 0  # index of system
    i_atom = 0  # position in descriptor array
    with open(folder / "function.data", "r") as f:
        for line in f:
            split = line.split()
            if len(split) == 1:
                # begin of system block
                i_atom_in_system = 0
                descriptor = np.zeros(
                    ((atoms_by_system[i_system]), dim * total_elems), dtype=float
                )  # need to explicitly cast as float to force conversion from string
            elif split == [
                "0.0000000000",
                "0.0000000000",
                "0.0000000000",
                "0.0000000000",
            ]:
                # end of system block
                i_system += 1
                descriptors.append(descriptor)
            else:
                # an actual output line!

                # sanity check
                z = int(split[0])  # charge of the atom we're looking at
                assert (
                    z == data.z[i_system][i_atom_in_system]
                ), f"Encountered irregularity while parsing ruNNer output! check {folder}"

                symmf = split[1::]

                # since the structure of symmfs varies by element type, we need
                # to separate them in the descriptor. here we do this by placing
                # the symmfs belonging to one element into blocks (rest stays zero)
                offset = dim * elems.index(z)
                # this truncates values if dim < len(symmf)
                realdim = min(dim, len(symmf))

                descriptor[i_atom_in_system][offset : offset + realdim] = symmf[0:realdim]
                i_atom_in_system += 1
                i_atom += 1

    # more (temporary) sanity checks
    assert (
        i_system == data.n
    ), f"Encountered irregularity while parsing ruNNer output! check {folder}"
    assert (
        len(descriptors) == data.n
    ), f"Encountered irregularity while parsing ruNNer output! check {folder}"
    assert (
        i_atom == total_atoms
    ), f"Encountered irregularity while parsing ruNNer output! check {folder}"

    return np.array(
        descriptors, dtype="O"
    )  # object ndarray for fancy indexing, it's really a list
