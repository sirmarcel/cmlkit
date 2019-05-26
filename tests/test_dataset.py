import numpy as np

from unittest import TestCase
import unittest.mock
import shutil
import pathlib

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
        self.splits = np.array([
                    [
                        np.random.randint(0, high=self.n, size=80),
                        np.random.randint(0, high=self.n, size=80),
                    ]
                    for i in range(3)
                ], dtype=object)

        self.data = Dataset(
            z=self.z,
            r=self.r,
            b=self.b,
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

    def test_creation(self):
        self.assertEqual(self.data.name, "test")
        self.assertEqual(self.data.desc, "test")
        np.testing.assert_array_equal(self.data.z, self.z)
        np.testing.assert_array_equal(self.data.r, self.r)
        np.testing.assert_array_equal(self.data.b, self.b)
        np.testing.assert_array_equal(self.data.p["p1"], self.p1)
        np.testing.assert_array_equal(self.data.p["p2"], self.p2)
        np.testing.assert_array_equal(self.data.splits, self.splits)

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
