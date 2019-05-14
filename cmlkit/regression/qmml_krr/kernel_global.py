"""Compute kernels between global representations.

This is just to provide a unified kernel interface.

"""

from cmlkit.engine import Component
from .kernel_functions import get_kernelf


class KernelGlobal(Component):
    """Compute kernels for global representations."""

    kind = "kernel_global"

    def __init__(self, kernelf, context={}):
        super().__init__(context=context)

        self.kernelf = get_kernelf(kernelf)

    def __call__(self, x, z=None):
        return self.kernelf(x=x, z=z)

    def _get_config(self):
        return {"kernelf": self.kernelf.get_config()}
