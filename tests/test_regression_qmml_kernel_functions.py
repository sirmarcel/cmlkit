from unittest import TestCase
import numpy as np
from functools import partial

from cmlkit.regression.qmml import (
    get_kernelf,
    KernelfGaussian,
    KernelfLaplacian,
    KernelfLinear,
)


def kernel_gaussian(x, ls, z=None):
    if z is None:
        z = x

    return np.exp(-np.sum((x - z) ** 2) / (2 * ls ** 2))


def kernel_laplacian(x, ls, z=None):
    if z is None:
        z = x

    return np.exp(-np.sum(np.abs(x - z)) / (ls))


def kernel_linear(x, z=None):
    if z is None:
        z = x

    return np.sum(x * z)


class TestKernelfs(TestCase):
    def setUp(self):
        self.x = np.random.random((1, 20))
        self.z = np.random.random((1, 20))

    def test_kernel_gaussian(self):
        kernel = KernelfGaussian(ls=2.0)
        np_kernel = partial(kernel_gaussian, ls=2.0)

        np.testing.assert_allclose(kernel(self.x), np_kernel(self.x))
        np.testing.assert_allclose(kernel(self.x, z=self.z), np_kernel(self.x, z=self.z))

    def test_kernel_gaussian_against_qmmlpack(self):
        kernel = KernelfGaussian(ls=2.0)
        from qmmlpack import kernel_gaussian as qmml_kernel

        np.testing.assert_allclose(kernel(self.x), qmml_kernel(self.x, theta=2.0))
        np.testing.assert_allclose(
            kernel(self.x, z=self.z), qmml_kernel(self.x, z=self.z, theta=2.0)
        )

        np.testing.assert_allclose(
            kernel(self.x, diagonal=True), qmml_kernel(self.x, diagonal=True, theta=2.0)
        )

    def test_kernel_laplacian(self):
        kernel = KernelfLaplacian(ls=2.0)
        np_kernel = partial(kernel_laplacian, ls=2.0)

        np.testing.assert_allclose(kernel(self.x), np_kernel(self.x))
        np.testing.assert_allclose(kernel(self.x, z=self.z), np_kernel(self.x, z=self.z))

    def test_kernel_linear(self):
        kernel = KernelfLinear()
        np_kernel = kernel_linear

        np.testing.assert_allclose(kernel(self.x), np_kernel(self.x))
        np.testing.assert_allclose(kernel(self.x, z=self.z), np_kernel(self.x, z=self.z))

    def test_deserialisation(self):
        self.assertEqual(
            get_kernelf({"gaussian": {"ls": 1.0}}).get_config(), {"gaussian": {"ls": 1.0}}
        )
        self.assertEqual(
            get_kernelf({"laplacian": {"ls": 1.0}}).get_config(),
            {"laplacian": {"ls": 1.0}},
        )
        self.assertEqual(get_kernelf({"linear": {}}).get_config(), {"linear": {}})
