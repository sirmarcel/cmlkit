from unittest import TestCase
import numpy as np

from cmlkit import Dataset
from cmlkit.representation.composed import Composed
from cmlkit.representation.mbtr import MBTR1


class TestComposed(TestCase):
    def setUp(self):
        self.data = Dataset(z=np.array([[0, 1]]), r=np.array([[[1, 2, 3], [1, 2, 3]]]))

    def test_mbtr_1(self):
        mbtr = MBTR1(
            start=0,
            stop=4,
            num=5,
            geomf="count",
            weightf="unity",
            broadening=0.001,
            eindexf="noreversals",
            aindexf="noreversals",
            elems=[0, 1, 2, 3],
            flatten=True,
        )

        composed = Composed(mbtr, mbtr.get_config())
        computed = composed(self.data)

        plain_computed = mbtr(self.data)

        np.testing.assert_array_equal(
            computed, np.concatenate((plain_computed, plain_computed), axis=1)
        )
