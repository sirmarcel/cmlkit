"""Compute kernels between atomic representations.

The kernel is computed by essentially taking the
underlying kernel functions between atomic representations
and summing up the resulting values for a given structure.

Naively, this leads to intermediate kernel matrices of dim

n_total_atoms x n_total_atoms,

which can be very large. To avoid this, we perform the computation
in blocks of a given `max_size`.

"""

import qmmlpack
from qmmlpack.experimental import recursive_matrix_map
import numpy as np

from cmlkit.engine import Component
from .kernel_functions import get_kernelf


class KernelAtomic(Component):
    """Compute kernels for atomic representations."""

    kind = "kernel_atomic"

    default_context = {"max_size": 256}

    def __init__(self, kernelf, norm=False, context={}):
        super().__init__(context=context)

        self.kernelf = get_kernelf(kernelf)
        self.norm = norm

        # self.max_size = self.context["max_size"]

    def __call__(self, x, z=None):
        return kernel_atomic(
            self.kernelf, x=x, z=z, norm=self.norm, max_size=self.context["max_size"]
        )

    def _get_config(self):
        return {"norm": self.norm, "kernelf": self.kernelf.get_config()}


def kernel_atomic(kernelf, x, z=None, norm=False, max_size=256):
    """Compute atomic kernel within a set of structures, or between two.

    Assumes atomic representation of fixed length dim. Each structure
    has n_atoms which vary across structures.

    Args:
        kernelf: Callable kernelf
        x: List (len = n_structures) of ndarrays with the atomic
                        representation along axis 1, i.e. n_atoms x dim arrays
        z: List, same as z. If specified, compute kernel between x and z.
        max_size: Integer, max block size for computation
        norm: Bool, if True the kernel values are normalised by n_atoms

    """

    if z is None:
        return _kernel_atomic(kernelf, x, x, symmetric=True, norm=norm, max_size=max_size)

    else:
        return _kernel_atomic(
            kernelf, x, z, symmetric=False, norm=norm, max_size=max_size
        )


def _kernel_atomic(kernelf, x, z, symmetric=False, norm=False, max_size=256):
    """Actual atomic kernel.

    Let n=len(x) and m=len(z).

    Each entry of the resulting n x m kernel matric is obtained by
    computing the kernel matrix between each atomic representation
    in each structure and summing up.

    This operation is performed for max_size x max_size chunks of
    the final kernel matrix.

    """

    def f(range_a, range_b):
        start_a, stop_a = range_a
        start_b, stop_b = range_b

        r_a = x[start_a:stop_a]
        r_b = z[start_b:stop_b]

        # Collect into contiguous chunks of memory
        # to efficiently compute kernelf
        x_a = np.concatenate(r_a, axis=0)
        x_b = np.concatenate(r_b, axis=0)

        k = kernelf(x_a, z=x_b)

        # Offsets are used to keep track of how to sum
        c_a, o_a = _get_counts_and_offsets(r_a)
        c_b, o_b = _get_counts_and_offsets(r_b)

        result = qmmlpack.partial_sum_matrix_reduce(k, o_a, indc=o_b)

        if norm:
            norms = np.outer(c_a, c_b)
            return result / norms
        else:
            return result

    return recursive_matrix_map(
        f, (len(x), len(z)), max_size=max_size, out=None, symmetric=symmetric
    )


def _get_counts_and_offsets(x):
    counts = np.array([len(s) for s in x], dtype=int)  # obtain n_atoms per structure

    # offsets: indices of beginning and end of representation belonging to structures
    offsets = np.zeros(len(counts) + 1, dtype=int)
    offsets[1::] = np.cumsum(
        counts
    )  # -> offsets = [0, n_atoms_1, n_atoms_1 + n_atoms_2, ...]

    return counts, offsets
