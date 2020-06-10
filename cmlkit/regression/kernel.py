from cmlkit import caches
from cmlkit.engine import Component

from .data import KernelMatrix


class Kernel(Component):
    """Base class for kernels."""

    # Sub-classes must provide kind

    def __init__(self, context={}):
        super().__init__(context=context)

    def __call__(self, x, z=None):

        if z is None:
            return KernelMatrix.from_array(self, x, self.compute_symmetric(x=x))
        else:

            return KernelMatrix.from_array(
                self, (x, z), self.compute_asymmetric(x=x, z=z)
            )
