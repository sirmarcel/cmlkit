from unittest import TestCase
import numpy as np

from cmlkit.regression.qmml import KernelGlobal
from cmlkit.regression.qmml import KernelfGaussian

kernelf = KernelfGaussian(ls=3.0)


class TestKernelGlobal(TestCase):
    def test_does_it_work(self):
        x = np.random.random((25, 10))
        kernel = KernelGlobal(kernelf=kernelf)

        np.testing.assert_allclose(kernel(x), kernelf(x))
