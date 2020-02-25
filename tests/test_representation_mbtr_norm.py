from unittest import TestCase
import numpy as np

from cmlkit import Dataset
from cmlkit.representation.mbtr.norm import SimpleNorm, NoneNorm, L2Norm


class TestSimpleNorm(TestCase):
    def test_does_normalise_to_1(self):
        fake_mbtr = np.random.random((5, 10))
        norm = SimpleNorm(scale=1.0)

        res = norm(fake_mbtr)

        self.assertAlmostEqual(np.sum(res[0]), np.sum(fake_mbtr[0]))
        self.assertAlmostEqual(np.sum(res[0]), 1.0)
        self.assertEqual(res.shape, fake_mbtr.shape)

    def test_does_normalise_to_not_1(self):
        fake_mbtr = np.random.random((5, 10))
        norm = SimpleNorm(scale=2.0)

        res = norm(fake_mbtr)

        self.assertAlmostEqual(np.sum(res[0]), 2.0)
        self.assertEqual(res.shape, fake_mbtr.shape)


class TestL2Norm(TestCase):
    def test_does_normalise_to_1(self):
        fake_mbtr = np.random.random((5, 10))
        norm = L2Norm(scale=1.0)

        res = norm(fake_mbtr)

        self.assertAlmostEqual(np.linalg.norm(res[0]), np.linalg.norm(fake_mbtr[0]))
        self.assertAlmostEqual(np.linalg.norm(res[0]), 1.0)
        self.assertEqual(res.shape, fake_mbtr.shape)

    def test_does_normalise_to_not_1(self):
        fake_mbtr = np.random.random((5, 10))
        norm = L2Norm(scale=2.0)

        res = norm(fake_mbtr)

        self.assertAlmostEqual(np.linalg.norm(res[0]), 2.0)
        self.assertEqual(res.shape, fake_mbtr.shape)


class TestNoneNorm(TestCase):
    def test_does_nothing(self):
        fake_mbtr = np.random.random((5, 10))
        norm = NoneNorm()

        res = norm(fake_mbtr)

        np.testing.assert_array_equal(res, fake_mbtr)
