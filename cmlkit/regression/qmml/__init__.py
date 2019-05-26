"""Kernel Ridge Regression as implemented in qmmlpack.

This provides a cmlkit interface to the underlying KRR
implementation on qmmlpack.

(Keeping in mind that cmlkit might support alternative KRR
implementations in the future!)
"""

from .kernel_functions import (
    get_kernelf,
    KernelfGaussian,
    KernelfLaplacian,
    KernelfLinear,
)

from .kernel_atomic import kernel_atomic, KernelAtomic
from .kernel_global import KernelGlobal
from .krr import KRR

components = [KernelAtomic, KernelGlobal, KRR]
