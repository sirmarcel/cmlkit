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

    def __call__(self, x):
        new_data = {"x": x.data["x"]*self.a}

        print(type(x))
        return DataExample.result(self, x, data=new_data)


cmlkit.register(DataExample, DataExampleComponent)


class TestDataBasic(TestCase):
    def setUp(self):
        self.tmpdir = (
            pathlib.Path(__file__) / ".."
        ).resolve() / "tmp_test_engine_data_data_basic"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_roundtrip(self):
        data = {"asdf": np.random.random(10), "jkl": np.random.random(10)}
        info = {"property": 123}

        data = DataExample.create(data=data, info=info)

        data.dump(self.tmpdir / "test")
        data2 = load_data(self.tmpdir / "test.npz")

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

    def test_same_consistent_id(self):
        new_data = self.component(self.startdata)
        new_data2 = self.component2(self.startdata2)
        self.assertEqual(new_data.id, new_data2.id)
        self.assertNotEqual(new_data.id, self.component3(self.startdata))

    def test_same_consistent_id_if_chained(self):
        new_data = self.component3(self.component(self.startdata))
        new_data2 = self.component3(self.component2(self.startdata2))
        self.assertEqual(new_data.id, new_data2.id)
