from unittest import TestCase
import time
import shutil
import pathlib
import numpy as np

from diskcache import Index
from concurrent.futures import TimeoutError, wait

import cmlkit
from cmlkit.engine import Component, parse_config
from cmlkit.utility import timed

from cmlkit.tune2.run.pool import EvaluationPool
from cmlkit.tune2.run.resultdb import ResultDB


class MockEvaluator(Component):
    kind = "mock_eval"

    def __call__(self, model):
        if model == {}:
            return {"loss": "hello"}
        elif model == "raise":
            raise ValueError("Hello!")
        elif model == "wait":
            time.sleep(0.2)
            return {"loss": "waited"}
        else:
            time.sleep(model["wait_for"])
            return {"loss": model["wait_for"]}

    def _get_config(self):
        return {}


cmlkit.register(MockEvaluator)


class TestEvaluationPool(TestCase):
    def setUp(self):
        self.tmpdir = pathlib.Path(__file__).parent / "tmp_test_soap"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_returns_correct_results_and_sets_cache(self):
        pool = EvaluationPool(
            max_workers=10,
            evaluator_config={"mock_eval": {}},
            caught_exceptions=(ValueError,),
            evals=ResultDB(
                Index(str(self.tmpdir / "test_returns_correct_results_and_sets_cache"))
            ),
        )

        future = pool.schedule({})
        self.assertEqual(pool.finish(future), {"ok": {"loss": "hello", "suggestion": {}}})
        self.assertEqual(
            pool.evals[future.eid], ["ok", {"loss": "hello", "suggestion": {}}]
        )

        # again!
        future = pool.schedule({})
        self.assertEqual(pool.finish(future), {"ok": {"loss": "hello", "suggestion": {}}})
        self.assertEqual(
            pool.evals[future.eid], ["ok", {"loss": "hello", "suggestion": {}}]
        )

    def test_raises_error_if_uncaught(self):
        pool = EvaluationPool(
            max_workers=10,
            evaluator_config={"mock_eval": {}},
            evals=ResultDB(Index(str(self.tmpdir / "test_raises_error_if_uncaught"))),
        )

        future = pool.schedule("raise")
        print(pool.caught_exceptions)
        with self.assertRaises(ValueError):
            pool.finish(future)

        # and it should not have done anything weird to the cache
        self.assertFalse(future.eid in pool.evals)

    def test_catches_exceptions(self):
        pool = EvaluationPool(
            max_workers=10,
            evaluator_config={"mock_eval": {}},
            caught_exceptions=(ValueError,),
            evals=ResultDB(Index(str(self.tmpdir / "test_catches_exceptions"))),
        )

        future = pool.schedule("raise")
        result = pool.finish(future)

        state, inner = parse_config(result)
        self.assertEqual(state, "error")
        self.assertEqual(inner["error"], "ValueError")
        self.assertEqual(inner["error_text"], "Hello!")
        self.assertTrue("traceback" in inner)

        # is the same found in the cache?
        state2, inner2 = pool.evals[future.eid]
        self.assertEqual(state2, "error")
        self.assertEqual(inner2, inner)

        # what if we do it again?
        future = pool.schedule("raise")
        result = pool.finish(future)

        state3, inner3 = pool.evals[future.eid]
        self.assertEqual(state3, "error")
        self.assertEqual(inner3, inner)

    def test_catches_timeout_exceptions(self):
        # this is a separate case because this exception
        # is raised at a slighly different location!
        pool = EvaluationPool(
            max_workers=10,
            evaluator_config={"mock_eval": {}},
            caught_exceptions=(TimeoutError,),
            evals=ResultDB(Index(str(self.tmpdir / "test_catches_timeout"))),
            trial_timeout=0.01,
        )

        future = pool.schedule("wait")
        result = pool.finish(future)

        state, inner = parse_config(result)
        self.assertEqual(state, "error")
        self.assertEqual(inner["error"], "TimeoutError")
        self.assertTrue("traceback" in inner)

        # is the same found in the cache?
        state2, inner2 = pool.evals[future.eid]
        self.assertEqual(state2, "error")
        self.assertEqual(inner2, inner)

        # what if we do it again?
        future = pool.schedule("wait")
        result = pool.finish(future)

        state3, inner3 = pool.evals[future.eid]
        self.assertEqual(state3, "error")
        self.assertEqual(inner3, inner)

    def test_basic_caching(self):
        pool = EvaluationPool(
            max_workers=10,
            evaluator_config={"mock_eval": {}},
            evals=ResultDB(Index(str(self.tmpdir / "test_basic_caching"))),
        )

        @timed
        def run_wait():
            future = pool.schedule("wait")
            result = pool.finish(future)
            state, inner = parse_config(result)

        res1, duration1 = run_wait()
        res2, duration2 = run_wait()

        self.assertGreater(duration1, duration2)
        self.assertEqual(res1, res2)

    def test_parallel_caching(self):
        # verify that the evalsdb can be written in parallel,
        # and that results are available in the end!
        pool = EvaluationPool(
            max_workers=10,
            evaluator_config={"mock_eval": {}},
            evals=ResultDB(Index(str(self.tmpdir / "test_parallel_caching"))),
        )

        times = np.random.random(20)
        futures = [pool.schedule({"wait_for": t}) for t in times]

        @timed
        def wait_for_finished():
            return wait(futures)

        res, duration = wait_for_finished()
        done, undone = res

        self.assertEqual(len(done), len(times))
        self.assertLess(duration, np.sum(times))  # did we achieve a speedup?
        self.assertEqual(len(pool.evals), len(times))
