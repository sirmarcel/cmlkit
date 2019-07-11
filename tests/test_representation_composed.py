from unittest import TestCase
import unittest.mock
import numpy as np
import shutil
import pathlib

from cmlkit import Dataset
from cmlkit.representation.composed import Composed
from cmlkit.representation.mbtr import MBTR1


class TestComposed(TestCase):
    def setUp(self):
        self.n = 100
        self.n_atoms = np.random.randint(1, high=10, size=self.n)

        r = [5 * np.random.random((na, 3)) for na in self.n_atoms]
        self.r = np.array(r, dtype=object)
        self.z = np.array(
            [np.random.randint(1, high=3, size=na) for na in self.n_atoms], dtype=object
        )

        self.data = Dataset(z=self.z, r=self.r)

        self.tmpdir = (pathlib.Path(__file__) / "..").resolve() / "tmp_test_composed"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

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
            elems=[1, 2, 3],
            flatten=True,
        )

        composed = Composed(mbtr, mbtr.get_config())
        computed = composed(self.data)

        plain_computed = mbtr(self.data)

        np.testing.assert_array_equal(
            computed, np.concatenate((plain_computed, plain_computed), axis=1)
        )

    def test_sf(self):
        with unittest.mock.patch.dict("os.environ", {"CML_SCRATCH": str(self.tmpdir)}):
            from cmlkit.representation.sf import SymmetryFunctions

            sf = SymmetryFunctions(
                elems=[1, 2, 3], universal=[{"rad_centered": {"n": 10, "cutoff": 6.0}}], context={"cleanup": False}
            )

            composed = Composed(sf, sf.get_config())
            computed = composed(self.data)

            single_manual = sf(self.data)

            for i in range(self.data.n):
                np.testing.assert_array_equal(
                    computed[i], np.concatenate([single_manual[i], single_manual[i]], axis=1)
                )
