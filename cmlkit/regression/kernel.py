from cmlkit.engine import Component

from .data import KernelMatrix


class Kernel(Component):
    """Base class for kernels."""

    # Sub-classes must provide kind

    def __init__(self, context={}):
        super().__init__(context=context)

    def __call__(self, x, z=None):

        if z is None:
            key = x.id
            result = self.cache.get_if_cached(key)
            if result is None:
                result = KernelMatrix.from_array(
                    self, x, self.compute_symmetric(x=x)
                )
                self.cache.submit(key, result)

            return result

        else:
            key = f"{x.id}+{z.id}"
            result = self.cache.get_if_cached(key)
            if result is None:
                result = KernelMatrix.from_array(
                    self, (x, z), self.compute_asymmetric(x=x, z=z)
                )

                self.cache.submit(key, result)

            return result
