"""Integration test for cache infra"""

from unittest import TestCase

import numpy as np
import shutil
import pathlib
import time

from cmlkit.engine import Component
from cmlkit.engine.data import Data

tmpdir = pathlib.Path(__file__).parent / "tmp_test_engine_cache"
tmpdir.mkdir(exist_ok=True)


class DummyComponent1(Component):
    kind = "dummy123"

    def __init__(self, a=1.0, context={}):
        super().__init__(context=context)

        self.a = a

    def _get_config(self):
        return {"a": self.a}

    def __call__(self, x):
        result = self.cache.get_if_cached(x.id)

        if result is None:
            result = Data.result(self, x, data={"y": self.compute(x.data["x"])})
            self.cache.submit(x.id, result)

        return result

    def compute(self, x):
        time.sleep(0.1)
        return x * self.a


class TestEngineCache(TestCase):
    def setUp(self):
        self.tmpdir = tmpdir
        self.tmpdir.mkdir(exist_ok=True)

        self.component = DummyComponent1(a=2.0)
        self.input = Data.create(data={"x": np.ones(3)})
        self.output = Data.create(
            data={"y": np.ones(3) * 2},
            history=[self.input.history[0], self.component.get_hid()],
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_nop_by_default(self):
        # does it return the correct result, and does it not go faster?

        start = time.monotonic()
        result = self.component(self.input)
        self.assertGreater(time.monotonic() - start, 0.1)
        print(result.history)
        print(self.output.history)
        self.assertEqual(
            result.get_config_hash(), self.output.get_config_hash()
        )

        start = time.monotonic()
        result = self.component(self.input)
        self.assertGreater(time.monotonic() - start, 0.1)
        self.assertEqual(
            result.get_config_hash(), self.output.get_config_hash()
        )

    def test_faster_with_diskcache(self):
        # does it return correct results,
        # and get faster?!

        component = DummyComponent1(
            a=2.0, context={"cache": {"disk": {"location": self.tmpdir}}}
        )

        start = time.monotonic()
        result = component(self.input)
        duration = time.monotonic() - start
        self.assertGreater(duration, 0.1)
        self.assertEqual(
            result.get_config_hash(), self.output.get_config_hash()
        )

        start = time.monotonic()
        result = component(self.input)
        self.assertGreater(duration, time.monotonic() - start)
        self.assertEqual(
            result.get_config_hash(), self.output.get_config_hash()
        )
