import numpy as np

from unittest import TestCase
import unittest.mock
import shutil
import pathlib
from copy import copy

from cmlkit.dataset import Dataset, Subset, load_dataset


class TestDataset(TestCase):
    def setUp(self):
        np.random.seed(123)

        self.tmpdir = pathlib.Path(__file__).parent / "tmp_test_dataset"
        self.tmpdir.mkdir(exist_ok=True)

        self.n = 100
        self.n_atoms = np.random.randint(1, high=10, size=self.n)

        r = [2 * np.random.random((na, 3)) for na in self.n_atoms]
        self.r = np.array(r, dtype=object)
        self.z = np.array(
            [np.random.randint(1, high=10, size=na) for na in self.n_atoms], dtype=object
        )
        self.b = np.random.random((self.n, 3, 3))
        self.p1 = np.random.random(self.n)
        self.p2 = np.random.random(self.n)
        self.splits = np.array(
            [
                [
                    np.random.randint(0, high=self.n, size=80),
                    np.random.randint(0, high=self.n, size=80),
                ]
                for i in range(3)
            ],
            dtype=object,
        )

        self.data = Dataset(
            z=self.z,
            r=self.r,
            b=self.b,
            p={"p1": self.p1, "p2": self.p2},
            name="test",
            desc="test",
            splits=self.splits,
        )

        self.data_nocell = Dataset(
            z=self.z,
            r=self.r,
            p={"p1": self.p1, "p2": self.p2},
            name="test",
            desc="test",
            splits=self.splits,
        )

        self.data2 = Dataset(
            z=self.z,
            r=self.r,
            b=self.b,
            p={"p1": self.p1, "p2": self.p2},
            name="test2",
            desc="test2",
        )

        self.data_nop = Dataset(
            z=self.z, r=self.r, b=self.b, p={}, name="test", desc="test"
        )

        # roll a new dataset

        n = 100
        n_atoms = np.random.randint(1, high=10, size=n)

        r = [2 * np.random.random((na, 3)) for na in n_atoms]
        r = np.array(r, dtype=object)
        z = np.array(
            [np.random.randint(1, high=10, size=na) for na in n_atoms], dtype=object
        )
        b = np.random.random((n, 3, 3))
        p1 = np.random.random(n)
        p2 = np.random.random(n)

        self.different = Dataset(
            z=z, r=r, b=b, p={"p1": p1, "p2": p2}, name="test_different", desc="test!"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_report(self):
        # smoke test
        self.data.report

    def test_creation(self):
        self.assertEqual(self.data.name, "test")
        self.assertEqual(self.data.desc, "test")
        np.testing.assert_array_equal(self.data.z, self.z)
        np.testing.assert_array_equal(self.data.r, self.r)
        np.testing.assert_array_equal(self.data.b, self.b)
        np.testing.assert_array_equal(self.data.p["p1"], self.p1)
        np.testing.assert_array_equal(self.data.p["p2"], self.p2)
        np.testing.assert_array_equal(self.data.splits, self.splits)

    def test_wrong_creation(self):
        with self.assertRaises(AssertionError):
            Dataset(z=self.data.z, r=self.data.r, p={"lol": [1]})

        with self.assertRaises(AssertionError):
            Dataset(z=[np.zeros(len(self.data.r[0]))], r=self.data.r)

    def test_hash_stable(self):
        # is the dataset hash stable across restarts?
        self.assertEqual(self.data.hash, "97e4cdce3be9851e9c109c3509bc65e1")

    def test_hash_equal(self):
        self.assertEqual(self.data.hash, self.data2.hash)
        self.assertEqual(self.data.geom_hash, self.data2.geom_hash)
        self.assertEqual(self.data.geom_hash, self.data_nop.geom_hash)
        self.assertTrue(self.data.hash is not None)
        self.assertTrue(self.data.geom_hash is not None)

        self.assertNotEqual(self.data.hash, self.different.hash)

    def test_roundtrip(self):
        self.data.save(directory=self.tmpdir)
        data3 = load_dataset("test", other_paths=[self.tmpdir])
        print(self.tmpdir)
        self.assertEqual(self.data.hash, data3.hash)
        self.assertEqual(self.data.name, data3.name)
        self.assertEqual(self.data.desc, data3.desc)
        np.testing.assert_array_equal(data3.splits, self.splits)

    def test_subset(self):
        idx = np.array([3, 1, 5, 6, 28, 32, 11], dtype=int)
        subset = Subset.from_dataset(self.data, idx=idx, name="subset")

        for i, index in enumerate(idx):
            np.testing.assert_array_equal(subset.z[i], self.data.z[index])
            np.testing.assert_array_equal(subset.b[i], self.data.b[index])
            np.testing.assert_array_equal(subset.r[i], self.data.r[index])
        self.assertEqual(subset.n, len(idx))

        # saving roundtrip test
        subset.save(directory=self.tmpdir)
        subset2 = load_dataset("subset", other_paths=[self.tmpdir])
        self.assertEqual(subset.hash, subset2.hash)

        # hash stability test
        self.assertEqual(subset.hash, "935443472f34bd24aa11d691f365c105")

    def test_chunking(self):
        for i, s in enumerate(self.data.in_chunks(size=30)):
            if i == 0:
                self.assertEqual(s.n, 30)
                np.testing.assert_array_equal(s.b, self.data.b[0:30])
            elif i == 1:
                self.assertEqual(s.n, 30)
                np.testing.assert_array_equal(s.b, self.data.b[30:60])
            elif i == 2:
                self.assertEqual(s.n, 30)
                np.testing.assert_array_equal(s.b, self.data.b[60:90])
            else:
                self.assertEqual(s.n, 10)
                np.testing.assert_array_equal(s.b, self.data.b[90:100])

    def test_ase(self):
        atoms = self.data.as_Atoms()

        for i, a in enumerate(atoms):
            np.testing.assert_array_equal(a.get_cell(), self.data.b[i])
            np.testing.assert_array_equal(a.get_positions(), self.data.r[i])
            np.testing.assert_array_equal(a.get_atomic_numbers(), self.data.z[i])

        dataset = Dataset.from_Atoms(atoms, p=self.data.p)
        self.assertEqual(dataset.geom_hash, self.data.geom_hash)
        for p in self.data.p.keys():
            np.testing.assert_array_equal(self.data.p[p], dataset.p[p])

        with self.assertRaises(AssertionError):
            atoms2 = copy(atoms)
            atoms2[0].set_pbc(False)
            Dataset.from_Atoms(atoms2)

        with self.assertRaises(AssertionError):
            atoms3 = copy(atoms)
            atoms3[0].set_pbc([False, True, False])
            Dataset.from_Atoms(atoms3)

        atoms = self.data_nocell.as_Atoms()

        for i, a in enumerate(atoms):
            np.testing.assert_array_equal(a.get_positions(), self.data_nocell.r[i])
            np.testing.assert_array_equal(a.get_atomic_numbers(), self.data_nocell.z[i])

        dataset = Dataset.from_Atoms(atoms)
        self.assertEqual(dataset.geom_hash, self.data_nocell.geom_hash)
