import os
import numpy as np
import time

import ase.io
import subprocess
from pathlib import Path
import shutil

from cmlkit import get_scratch, quippy_pythonpath, quippy_python_exe
from cmlkit.engine import compute_hash, parse_config
from cmlkit.utility import charges_to_elements


quippy_execute = Path(__file__).parents[0] / "quippy_execute.py"


def compute_soap(data, config, cleanup=True, timeout=None):
    """Compute the SOAP representation.

    Actually computes the SOAP using the quippy code. It does it by

    a) Writing the data to a scratch partition (with ase).
    b) Calling out to a sub-process, which runs the `quippy_execute.py`
       script as a command.
    c) ... which reads the data, and actuallyc computes SOAP, and writes it out.
    d) Then we read it back in the main process.

    This is clearly insanely inefficient, so why do we do it?
    Quippy is at the moment only available in Python 2.7. `cmlkit` is aggressively
    not Python 2.7 compatible. So we need to open a whole new process to run the
    old Python interpreter...

    Args:
        data: Dataset instance
        config: dict containing the keys
            'sigma': broadening
            'n_max': number of radial basis functions
            'l_max': number of angular basis functions
            'cutoff': cutoff radius
            'elems': elements to be computed
            (for more, see the `config` file.)
        timeout: maximum number of seconds to wait for computations
        cleanup: if True, delete scratch files

    Returns:
        Computed (atomic) representation, i.e. an ndarray-wrapped list with each entry
        being an array with the representations for each atom in that structure.

    """

    quippy_config = make_quippy_config(config)
    folder = prepare_task(data, quippy_config)
    stdout, stderr = run_task(folder, timeout=timeout)
    result = read_result(folder)

    if cleanup:
        shutil.rmtree(folder)

    return result


def check_dependencies():
    assert (
        quippy_pythonpath is not None
    ), f"Could not find $CML_QUIPPY_PYTHONPATH, which should contain quippy."
    assert (
        quippy_python_exe is not None
    ), f"Could not find $CML_QUIPPY_PYTHON_EXE, which should point to a python 2.7 executable."


def make_quippy_config(config):
    """Generate the quippy descriptor argument string."""

    # doing this here to make the f-string slightly less horrible
    cutoff = config["cutoff"]
    l_max = config["l_max"]
    n_max = config["n_max"]
    sigma = config["sigma"]
    elems = config["elems"]

    species = " ".join(map(str, elems))
    quippy_config = f"soap cutoff={cutoff} l_max={l_max} n_max={n_max} atom_sigma={sigma} n_Z={len(elems)} Z={{{species}}} n_species={len(elems)} species_Z={{{species}}}"

    return quippy_config


def prepare_task(data, quippy_config):
    tid = compute_hash(time.time(), np.random.rand(), data.geom_hash, quippy_config)
    folder = get_scratch() / f"soap_{tid}"
    folder.mkdir(parents=True)

    write_data(data, folder)

    with open(folder / "quippy_config.txt", "w+") as f:
        f.write(quippy_config)

    return folder


def write_data(data, folder):
    ase.io.write(str(folder / "data.traj"), data.as_Atoms(), format="traj", parallel=False)


def run_task(folder, timeout=None):
    env = {
        "PYTHONPATH": quippy_pythonpath,  # environment containing quippy and ase
        "HOME": os.environ["HOME"],
    }

    finished = subprocess.run(
        [quippy_python_exe, quippy_execute],
        cwd=folder,
        timeout=timeout,
        # check=True,
        encoding="utf-8",
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )

    if "Error" in finished.stderr or "Error" in finished.stdout:
        # TODO: make more useful errors
        raise Exception(
            f"SOAP did not terminate correctly. Here is stderr:\n{finished.stderr}\nHere is stdout:\n{finished.stdout}"
        )

    return finished.stdout, finished.stderr


def read_result(folder):
    return np.load(
        folder / "out.npy", fix_imports=True, encoding="bytes", allow_pickle=True
    )
