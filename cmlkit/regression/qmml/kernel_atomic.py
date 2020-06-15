"""Compute kernels between atomic representations.

The kernel is computed by essentially taking the
underlying kernel functions between atomic representations
and summing up the resulting values for a given structure.

Naively, this leads to intermediate kernel matrices of dim

n_total_atoms x n_total_atoms,

which can be very large. To avoid this, we perform the computation
in blocks of a given `max_size`.

"""

import numpy as np

from cmlkit.engine import Component
from cmlkit.utility import import_qmmlpack, import_qmmlpack_experimental
from cmlkit.representation.data import AtomicRepresentation

from cmlkit.regression import Kernel

from .kernel_functions import get_kernelf


class KernelAtomic(Kernel):
    """Compute kernels for atomic representations."""

    kind = "kernel_atomic"

    default_context = {"max_size": 256}

    def __init__(self, kernelf, norm=False, context={}):
        super().__init__(context=context)

        self.kernelf = get_kernelf(kernelf)
        self.norm = norm

    def compute_symmetric(self, x):
        assert isinstance(
            x, AtomicRepresentation
        ), "KernelAtomic only works on atomic representations."
        return _kernel_atomic(
            self.kernelf,
            x=x,
            z=x,
            symmetric=True,
            norm=self.norm,
            max_size=self.context["max_size"],
        )

    def compute_asymmetric(self, x, z):
        assert isinstance(
            x, AtomicRepresentation
        ), "KernelAtomic only works on atomic representations."
        assert isinstance(
            z, AtomicRepresentation
        ), "KernelAtomic only works on atomic representations."
        return _kernel_atomic(
            self.kernelf,
            x=x,
            z=z,
            symmetric=False,
            norm=self.norm,
            max_size=self.context["max_size"],
        )

    def _get_config(self):
        return {"norm": self.norm, "kernelf": self.kernelf.get_config()}


def kernel_atomic(kernelf, x, z=None, norm=False, max_size=256):
    """Compute atomic kernel within a set of atomic representations, or between two.

    Convenience wrapper for lower-level functions. See _kernel_atomic
    for implementation details.

    Args:
        kernelf: Callable kernelf
        x: AtomicRepresentation
        z: AtomicRepresentation
        max_size: Integer, max block size for computation
        norm: Bool, if True the kernel values are normalised by n_atoms

    """

    if z is None:
        return _kernel_atomic(
            kernelf, x, x, symmetric=True, norm=norm, max_size=max_size
        )

    else:
        return _kernel_atomic(
            kernelf, x, z, symmetric=False, norm=norm, max_size=max_size
        )


def _kernel_atomic(kernelf, x, z, symmetric=False, norm=False, max_size=256):
    """Atomic kernel.

    Let n=len(x) and m=len(z).

    Each entry of the resulting n x m kernel matric is obtained
    as follows:

        For i in n and j in m:

        K_ij = sum( kernelf(x[i], y[i]) )

    In practical terms, this means the kernel matrix entry between
    two structures is simply the sum over all entries of the kernel
    matrix between the representations of each atom in each structures.

    To avoid computing an intermediate n_atoms_x x n_atoms_z kernel
    matrix, the computation is done in max_size x max_size chunks of
    the final kernel matrix.

    The heavy lifting here is done by low-level qmmlpack functions.
    recursive_matrix_map picks chunks of the final kernel matrix
    to compute, respecting the chunk size. We then take out the
    sections of the linearised atomic representations, compute the
    "atom-atom" kernel matrix for the entire chunk (which contains
    multiple systems!) and pass it to partial_sum_matrix_reduce,
    which takes care of summing up by system.

    """
    qmmlpack = import_qmmlpack("use cmlkit.regression.qmml")
    experimental = import_qmmlpack_experimental("use cmlkit.regression.qmml")

    def f(range_x, range_z):
        this_x = x.range(range_x)
        this_z = z.range(range_z)

        k = kernelf(this_x.linear, z=this_z.linear)

        result = qmmlpack.partial_sum_matrix_reduce(
            k, this_x.offsets, indc=this_z.offsets
        )

        if norm:
            norms = np.outer(this_x.counts, this_z.counts)
            return result / norms
        else:
            return result

    return experimental.recursive_matrix_map(
        f, (x.n, z.n), max_size=max_size, out=None, symmetric=symmetric
    )
