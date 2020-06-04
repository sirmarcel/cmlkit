from cmlkit.engine import Data, load_data
import cmlkit

import numpy as np
from unittest import TestCase
import unittest.mock
import shutil
import pathlib


class DataExample(Data):

    kind = "test_data"
    version = 1


cmlkit.register(DataExample)


class TestData(TestCase):
    def setUp(self):
        self.tmpdir = (
            pathlib.Path(__file__) / ".."
        ).resolve() / "tmp_test_engine_data_data"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_roundtrip(self):
        data = {"asdf": np.random.random(10), "jkl": np.random.random(10)}
        info = {"property": 123}

        data = DataExample(data=data, info=info)

        data.dump(self.tmpdir / "test")
        data2 = load_data(self.tmpdir / "test.npz")

        np.testing.assert_array_equal(data.data["asdf"], data2.data["asdf"])
        np.testing.assert_array_equal(data.data["jkl"], data2.data["jkl"])
        self.assertEqual(data.info["property"], data2.info["property"])
        self.assertEqual(data.version, data2.version)
