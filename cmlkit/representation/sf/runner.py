"""Compute symmetry functions with RuNNer."""

import shutil

from .runner_input import prepare_task
from .runner_output import run_task, read_result

from cmlkit import runner_path


def compute_symmfs(data, config, timeout=None, cleanup=True):
    """Compute atom-centred symmetry functions

    Perform the actual computation of symmetry functions using the ruNNer backend.

    Args:
        data: Dataset instance
        config: dict containing the keys
            'universal': configs of universal SFs
            'elemental': configs of elemental SFs
            'dim': dimensionality of descriptor, the maximum number of symmetry
                function values per element (will be truncated silently!)
            'elems': elements to be computed
            (for more, see the `config` file.)
        timeout: maximum number of seconds to wait for computations
        cleanup: if True, delete scratch files

    Returns:
        Computed (atomic) representation, i.e. an ndarray-wrapped list with each entry
        being an array with the representations for each atom in that structure.
    """

    check_dependencies()
    folder = prepare_task(data, config)
    out, err = run_task(folder, timeout=timeout)
    descriptors = read_result(data, folder, config)

    if cleanup:
        shutil.rmtree(folder)

    return descriptors


def check_dependencies():
    assert (
        runner_path is not None
    ), "Cannot find RuNNer executable, which should be specified as $CML_RUNNER_PATH."
