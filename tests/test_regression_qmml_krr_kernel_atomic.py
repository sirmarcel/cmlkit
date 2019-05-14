from unittest import TestCase
import numpy as np

from cmlkit.regression.qmml_krr import kernel_atomic, KernelAtomic
from cmlkit.regression.qmml_krr import KernelfGaussian

kernelf = KernelfGaussian(ls=3.0)


class TestKernelAtomicObject(TestCase):
    def test_does_it_work(self):
        counts_x = [5, 3, 1]
        x = [np.random.random((i, 10)) for i in counts_x]
        kernel = KernelAtomic(kernelf=kernelf, norm=True)

        result = kernel(x)

        reference_result = np.array(
            [
                [
                    np.sum(kernelf(x=sys, z=sys2))
                    / (counts_x[i] * counts_x[i2])
                    for i2, sys2 in enumerate(x)
                ]
                for i, sys in enumerate(x)
            ]
        )

        np.testing.assert_allclose(result, reference_result)


class TestKernelAtomicFunction(TestCase):
    def setUp(self):
        self.counts_x = [5, 3, 1]
        self.counts_z = [1, 2]

        self.x = [np.random.random((i, 10)) for i in self.counts_x]
        self.z = [np.random.random((i, 10)) for i in self.counts_z]

    def test_atomic_kernel_one(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, norm=False)
        reference_result = np.array(
            [[np.sum(kernelf(x=sys, z=sys2)) for sys2 in self.x] for sys in self.x]
        )

        np.testing.assert_allclose(result, reference_result)

    def test_atomic_kernel_one_norm(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, norm=True)
        reference_result = np.array(
            [
                [
                    np.sum(kernelf(x=sys, z=sys2))
                    / (self.counts_x[i] * self.counts_x[i2])
                    for i2, sys2 in enumerate(self.x)
                ]
                for i, sys in enumerate(self.x)
            ]
        )

        np.testing.assert_allclose(result, reference_result)

    def test_atomic_kernel_two(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, z=self.z, norm=False)
        reference_result = np.array(
            [[np.sum(kernelf(x=sys, z=sys2)) for sys2 in self.z] for sys in self.x]
        )

        np.testing.assert_allclose(result, reference_result)

    def test_atomic_kernel_two_norm(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, z=self.z, norm=True)
        reference_result = np.array(
            [
                [
                    np.sum(kernelf(x=sys, z=sys2))
                    / (self.counts_x[i] * self.counts_z[i2])
                    for i2, sys2 in enumerate(self.z)
                ]
                for i, sys in enumerate(self.x)
            ]
        )

        np.testing.assert_allclose(result, reference_result)
