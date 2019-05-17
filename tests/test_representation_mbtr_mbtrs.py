from unittest import TestCase
import numpy as np

from cmlkit import Dataset
from cmlkit.representation.mbtr import MBTR1, MBTR2, MBTR3


class TestMBTR1(TestCase):
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

        computed = mbtr(self.data)

        # => [[0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
        # which is in non-flattened form:
        # [[[0. 1. 0. 0. 0.]
        #   [0. 1. 0. 0. 0.]
        #   [0. 0. 0. 0. 0.]
        #   [0. 0. 0. 0. 0.]]]
        # The count is one (i=1), once (value=1), for
        # the elements 0 and 1. (first row, second row.)

        self.assertEqual(computed[0][1], 1.0)
        self.assertEqual(computed[0][6], 1.0)


class TestMBTR2(TestCase):
    def setUp(self):
        self.data = Dataset(
            z=np.array([[0, 0]]), r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        )

    def test_mbtr_2(self):
        mbtr = MBTR2(
            start=0,
            stop=1,
            num=3,
            geomf="1/distance",
            weightf="unity",
            broadening=0.001,
            eindexf="noreversals",
            aindexf="noreversals",
            elems=[0],
            flatten=True,
        )

        computed = mbtr(self.data)
        print(computed)

        # => [[0. 1.0 0.]]
        # There is one inverse distance: 0.5, in the middle of
        # the discretisation interval.

        self.assertEqual(computed[0][1], 1.0)
        self.assertEqual(computed[0][2], 0.0)

    def test_mbtr_2_with_parametrized_weightf(self):
        # just make sure things don't explode
        mbtr = MBTR2(
            start=0,
            stop=1,
            num=3,
            geomf="1/distance",
            weightf={"exp_-1/identity": {"ls": 1.0}},
            broadening=0.001,
            eindexf="noreversals",
            aindexf="noreversals",
            elems=[0],
            flatten=True,
        )

        computed = mbtr(self.data)


class TestMBTR3(TestCase):
    def setUp(self):
        self.data = Dataset(
            z=np.array([[0, 0, 0]]),
            r=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        )

    def test_mbtr_3(self):
        mbtr = MBTR3(
            start=0,
            stop=np.pi / 2 + 0.01,
            num=3,
            geomf="angle",
            weightf="unity",
            broadening=0.001,
            eindexf="noreversals",
            aindexf="noreversals",
            elems=[0],
            flatten=True,
        )

        computed = mbtr(self.data)

        # => [[0. 2. 1.]]
        # There are three angles: once 90 degrees,
        # then twice 45 degrees.

        self.assertEqual(computed[0][1], 2.0)
        self.assertEqual(computed[0][2], 1.0)
