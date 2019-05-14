"""Kernel functions as implemented by qmmlpack.

A kernelf function, or kernelf for short is *not* the
same as a kernel in all cases. In `cmlkit` terminology, the
kernel is the whole procedure of computing distances between
two systems. So for global representations, the kernelf is
often the kernel, but when atomic representations are used,
we typically compute a kernel function between these representations
individually and combine the result into an overall kernel.

This is really just a thin translation layer that makes the qmmlpack
callables de/serisalisable in the `cmlkit` way.

There is no getattr magic because it doesn't seem needed.

"""

import qmmlpack

from cmlkit.engine import Component, _from_config


def get_kernelf(config, context={}):
    """Get a kernel function."""
    return _from_config(config, classes=classes, context=context)


def get_raw_kernelf(name):
    """Get the raw callable for a kernel function."""
    # this is in anticipation of having to write "tune" KRR classes
    return kernelfs[name]


class Kernelf(Component):
    """Base class for kernel functions."""

    def __init__(self, ls, context={}):
        super().__init__(context=context)

        self.ls = ls

    def __call__(self, x, z=None, diagonal=False, distance=False):
        kernelf = kernelfs[self.kind]
        return kernelf(x=x, z=z, theta=self.ls, diagonal=False, distance=False)

    def _get_config(self):
        return {"ls": self.ls}


class KernelfGaussian(Kernelf):
    """Gaussian Kernel Function.

    Gaussian kernel k(x,z) = exp(-||x-z||^2/2s^2).

    For further information, see qmmlpack docs.

    Attributes:
        ls: length scale (s in formula)

    """

    kind = "gaussian"


class KernelfLaplacian(Kernelf):
    """Laplacian Kernel Function.

    Laplacian kernel k(x,z) = exp(-||x-z||_1/s).

    For further information, see qmmlpack docs.

    Attributes:
        ls: length scale (s in formula)
    """

    kind = "laplacian"


class KernelfLinear(Kernelf):
    """Linear Kernel Function.

    Linear kernel k(x,z) = <x,z>

    """

    kind = "linear"

    def __init__(self, context={}):
        super().__init__(ls=None, context=context)

    def __call__(self, x, z=None, diagonal=False):
        kernelf = kernelfs[self.kind]
        return kernelf(x=x, z=z, theta=self.ls, diagonal=False)

    def _get_config(self):
        return {}


kernelfs = {
    "gaussian": qmmlpack.kernel_gaussian,
    "laplacian": qmmlpack.kernel_laplacian,
    "linear": qmmlpack.kernel_linear,
}

classes = {
    "gaussian": KernelfGaussian,
    "laplacian": KernelfLaplacian,
    "linear": KernelfLinear,
}
