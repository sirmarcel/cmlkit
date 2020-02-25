# Does anything work at all?

from unittest import TestCase
import pathlib
import shutil
import time

from cmlkit.engine import diskcached, memcached
from cmlkit.utility import timed


def slow_f(x):
    time.sleep(0.5)
    return 2 * x


class TestDiskCache(TestCase):
    def setUp(self):
        self.tmpdir = (
            pathlib.Path(__file__) / ".."
        ).resolve() / "tmp_test_engine_caching"
        self.tmpdir.mkdir(exist_ok=True)

        self.f = diskcached(slow_f, cache_location=self.tmpdir, min_duration=0.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_disk(self):
        a = self.f(2)
        self.assertEqual(a, 4)

        # check if we have actually written something
        with self.assertRaises(OSError):
            self.tmpdir.rmdir()

        # check if we achieved a speedup
        f = timed(self.f)
        b, t = f(2)
        self.assertLess(t, 0.5)


class TestMemCache(TestCase):
    def setUp(self):
        self.f = memcached(slow_f)

    def test_disk(self):
        a = self.f(2)
        self.assertEqual(a, 4)

        # check if we achieved a speedup
        f = timed(self.f)
        b, t = f(2)
        self.assertLess(t, 0.5)
