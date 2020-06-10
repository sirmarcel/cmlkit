"""Compute kernels between global representations.

This is just to provide a unified kernel interface.

"""

from cmlkit.regression import Kernel
from cmlkit.representation.data import GlobalRepresentation
from .kernel_functions import get_kernelf


class KernelGlobal(Kernel):
    """Compute kernels for global representations."""

    kind = "kernel_global"

    def __init__(self, kernelf, context={}):
        super().__init__(context=context)
        self.kernelf = get_kernelf(kernelf)

    def compute_symmetric(self, x, z=None):
        assert isinstance(
            x, GlobalRepresentation
        ), "KernelGlobal only works on global representations."
        return self.kernelf(x=x.array)

    def compute_asymmetric(self, x, z):
        assert isinstance(
            x, GlobalRepresentation
        ), "KernelGlobal only works on global representations."
        assert isinstance(
            z, GlobalRepresentation
        ), "KernelGlobal only works on global representations."
        return self.kernelf(x=x.array, z=z.array)

    def _get_config(self):
        return {"kernelf": self.kernelf.get_config()}
