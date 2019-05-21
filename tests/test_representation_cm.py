from unittest import TestCase
import numpy as np

from cmlkit import Dataset
from cmlkit.representation.coulomb_matrix import CoulombMatrix


class TestCM(TestCase):
    def setUp(self):
        self.data = Dataset(z=np.array([[2, 3]]), r=np.array([[[0, 0, 0], [2.0, 0, 0]]]))

    def test_work(self):
        cm = CoulombMatrix(padding=3, unit="BohrRadius", flatten=True)

        computed = cm(self.data)
        print(computed)

        np.testing.assert_almost_equal(computed[0][0], 0.5 * 3.0 ** (2.4))
        np.testing.assert_almost_equal(computed[0][1], 2 * 3 / 2.0)
        np.testing.assert_almost_equal(computed[0][2], 0.5 * 2.0 ** (2.4))
        np.testing.assert_almost_equal(computed[0][3], 0)
