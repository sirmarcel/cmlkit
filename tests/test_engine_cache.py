"""Integration test for cache infra"""

from unittest import TestCase

import shutil
import pathlib
import time

from cmlkit.engine import Component, compute_hash
from cmlkit.engine.cache import HyperCache

tmpdir = pathlib.Path(__file__).parent / "tmp_test_engine_cache"
tmpdir.mkdir(exist_ok=True)

hypercache = HyperCache(location=tmpdir)


class DummyComponent(Component):
    kind = "dummy123"

    def __init__(self, a=1.0, context={}):
        super().__init__(context=context)

        self.a = a

        self.cache = hypercache.register(self)

    def _get_config(self):
        return {"a": self.a}

    def __call__(self, x):

        if self.cache:
            key = compute_hash(x)
            result = self.cache.get_if_cached(key)

            if result is None:
                result = self.compute(x)
                self.cache.submit(key, result)

        else:
            result = self.compute(x)

        return result

    def compute(self, x):
        time.sleep(0.1)
        return x * self.a


class TestEngineCache(TestCase):
    def setUp(self):
        self.tmpdir = tmpdir
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_nop_by_default(self):
        component = DummyComponent(a=2.0)

        start = time.monotonic()
        result = component(1.0)
        self.assertEqual(result, 2.0)
        self.assertGreater(time.monotonic() - start, 0.1)

        start = time.monotonic()
        result = component(1.0)
        self.assertEqual(result, 2.0)
        self.assertGreater(time.monotonic() - start, 0.1)

        start = time.monotonic()
        result = component(2.0)
        self.assertEqual(result, 4.0)
        self.assertGreater(time.monotonic() - start, 0.1)

    def test_faster_with_diskcache(self):
        component = DummyComponent(a=2.0, context={"cache": "disk"})

        start = time.monotonic()
        result = component(1.0)
        self.assertEqual(result, 2.0)
        self.assertGreater(time.monotonic() - start, 0.1)

        start = time.monotonic()
        result = component(1.0)
        self.assertEqual(result, 2.0)
        self.assertLess(time.monotonic() - start, 0.1)

        # sanity check that it is sensitive to actual input
        start = time.monotonic()
        result = component(2.0)
        self.assertEqual(result, 4.0)
        self.assertGreater(time.monotonic() - start, 0.1)
