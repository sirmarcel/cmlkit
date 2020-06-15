from unittest import TestCase
import numpy as np

from cmlkit.regression.data import KernelMatrix
from cmlkit.representation.data import AtomicRepresentation

from cmlkit.regression.qmml import kernel_atomic, KernelAtomic
from cmlkit.regression.qmml import KernelfGaussian

kernelf = KernelfGaussian(ls=3.0)


class TestKernelAtomicObject(TestCase):
    def test_does_it_work(self):
        counts = [5, 3, 1, 4, 5]
        x = AtomicRepresentation.mock(counts, np.random.random((18, 10)))

        kernel = KernelAtomic(kernelf=kernelf, norm=True, context={"max_size": 4})

        result = kernel(x)

        reference_result = np.array(
            [
                [
                    np.sum(kernelf(x=sys, z=sys2))
                    / (counts[i] * counts[i2])
                    for i2, sys2 in enumerate(x.ragged)
                ]
                for i, sys in enumerate(x.ragged)
            ]
        )

        np.testing.assert_allclose(result.array, reference_result)


class TestKernelAtomicFunction(TestCase):
    def setUp(self):
        self.counts_x = [5, 3, 1, 4, 2, 3, 2]
        self.counts_z = [1, 2, 3, 3, 1]

        self.x = AtomicRepresentation.mock(self.counts_x, np.random.random((np.sum(self.counts_x), 10)))
        self.z = AtomicRepresentation.mock(self.counts_z, np.random.random((np.sum(self.counts_z), 10)))

    def test_atomic_kernel_one(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, norm=False)
        reference_result = np.array(
            [[np.sum(kernelf(x=sys, z=sys2)) for sys2 in self.x.ragged] for sys in self.x.ragged]
        )

        np.testing.assert_allclose(result, reference_result)

    def test_atomic_kernel_one_norm(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, norm=True)
        reference_result = np.array(
            [
                [
                    np.sum(kernelf(x=sys, z=sys2))
                    / (self.counts_x[i] * self.counts_x[i2])
                    for i2, sys2 in enumerate(self.x.ragged)
                ]
                for i, sys in enumerate(self.x.ragged)
            ]
        )

        np.testing.assert_allclose(result, reference_result)

    def test_atomic_kernel_two(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, z=self.z, norm=False, max_size=4)
        reference_result = np.array(
            [[np.sum(kernelf(x=sys, z=sys2)) for sys2 in self.z.ragged] for sys in self.x.ragged]
        )

        np.testing.assert_allclose(result, reference_result)

    def test_atomic_kernel_two_norm(self):
        result = kernel_atomic(kernelf=kernelf, x=self.x, z=self.z, norm=True, max_size=4)
        reference_result = np.array(
            [
                [
                    np.sum(kernelf(x=sys, z=sys2))
                    / (self.counts_x[i] * self.counts_z[i2])
                    for i2, sys2 in enumerate(self.z.ragged)
                ]
                for i, sys in enumerate(self.x.ragged)
            ]
        )

        np.testing.assert_allclose(result, reference_result)
