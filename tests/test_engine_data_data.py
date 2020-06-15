from cmlkit.engine import Data, load_data, Component
import cmlkit

import numpy as np
from unittest import TestCase
import unittest.mock
import shutil
import pathlib


class DataExample(Data):

    kind = "test_data"


class DataExampleComponent(Component):

    kind = "test_data_component"

    def __init__(self, a, context={}):
        super().__init__(context=context)

        self.a = a

    def _get_config(self):
        return {"a": self.a}

    def __call__(self, x, y=None):
        if y is None:
            new_data = {"x": x.data["x"]*self.a}
            return DataExample.result(self, x, data=new_data)
        else:
            new_data = {"x": x.data["x"]*self.a+y.data["x"]*self.a}
            return DataExample.result(self, (x, y), data=new_data)


cmlkit.register(DataExample, DataExampleComponent)


class TestDataBasic(TestCase):
    def setUp(self):
        self.tmpdir = (
            pathlib.Path(__file__) / ".."
        ).resolve() / "tmp_test_engine_data_data_basic"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_roundtrip_protocol_1(self):
        data = {"asdf": np.random.random(10), "jkl": np.random.random(10)}
        info = {"property": 123}

        data = DataExample.create(data=data, info=info)

        data.dump(self.tmpdir / "test_1", protocol=1)
        data2 = load_data(self.tmpdir / "test_1.npz")

        np.testing.assert_array_equal(data.data["asdf"], data2.data["asdf"])
        np.testing.assert_array_equal(data.data["jkl"], data2.data["jkl"])
        self.assertEqual(data.info["property"], data2.info["property"])
        self.assertEqual(data.history, data2.history)
        self.assertEqual(data.id, data2.id)

    def test_roundtrip_protocol_1(self):
        data = {"asdf": np.random.random(10), "jkl": np.random.random(10)}
        info = {"property": 123}

        data = DataExample.create(data=data, info=info)

        data.dump(self.tmpdir / "test_2", protocol=2)
        data2 = load_data(self.tmpdir / "test_2.npz")

        np.testing.assert_array_equal(data.data["asdf"], data2.data["asdf"])
        np.testing.assert_array_equal(data.data["jkl"], data2.data["jkl"])
        self.assertEqual(data.info["property"], data2.info["property"])
        self.assertEqual(data.history, data2.history)
        self.assertEqual(data.id, data2.id)


class TestDataTracking(TestCase):
    def setUp(self):
        self.component = DataExampleComponent(a=1.0)
        self.component2 = DataExampleComponent(a=1.0)
        self.component3 = DataExampleComponent(a=2.0)
        self.startdata = DataExample.create(data={"x": 1.0})
        self.startdata2 = DataExample.create(data={"x": 1.0})

        self.other_data = DataExample.create(data={"x": 2.0})

    def test_same_consistent_id(self):
        new_data = self.component(self.startdata)
        new_data2 = self.component2(self.startdata2)
        self.assertEqual(new_data.id, new_data2.id)
        self.assertNotEqual(new_data.id, self.component3(self.startdata))

    def test_same_consistent_id_if_chained(self):
        new_data = self.component3(self.component(self.startdata))
        new_data2 = self.component3(self.component2(self.startdata2))
        self.assertEqual(new_data.id, new_data2.id)

    def test_same_consistent_id_if_combined(self):
        new_data = self.component(self.startdata, y=self.other_data)
        new_data2 = self.component(self.startdata2, y=self.other_data)

        self.assertEqual(new_data.id, new_data2.id)

        # another little smoke test to see if the hashes are
        # really consistent. if this fails, file an issue please!
        self.assertEqual(new_data.id, "5f97bec5fc0d84be6faf50a2a0d6d025")

    def test_different_id_if_combined_in_different_order(self):
        new_data = self.component(self.startdata, y=self.other_data)
        new_data2 = self.component(self.other_data, y=self.startdata)
        self.assertNotEqual(new_data.id, new_data2.id)
