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
from cmlkit.tune2.search.hyperopt import Hyperopt


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


class MockEvaluator2(Component):
    kind = "mock_eval2"

    def __call__(self, model):
        time.sleep(model["wait"])
        loss = (
            (model["x"] - 2.0) ** 2
            + (model["y"] - 1.0) ** 2
            + (model["z"] - 0.0) ** 2
            + (model["a"] + 1.0) ** 2
            + (model["b"] + 2.0) ** 2
            + (model["c"] + 3.0) ** 2
        )
        return {"loss": loss}

    def _get_config(self):
        return {}


cmlkit.register(MockEvaluator, MockEvaluator2)


class TestEvaluationPoolWithCache(TestCase):
    def setUp(self):
        self.tmpdir = pathlib.Path(__file__).parent / "tmp_test_pool"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_returns_correct_results_and_sets_cache(self):
        pool = EvaluationPool(
            max_workers=4,
            evaluator_config={"mock_eval": {}},
            caught_exceptions=(ValueError,),
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
            max_workers=4,
            evaluator_config={"mock_eval": {}},
        )

        future = pool.schedule("raise")
        print(pool.caught_exceptions)
        with self.assertRaises(ValueError):
            pool.finish(future)

        # and it should not have done anything weird to the cache
        self.assertFalse(future.eid in pool.evals)
        pool.shutdown()

    def test_catches_exceptions(self):
        pool = EvaluationPool(
            max_workers=4,
            evaluator_config={"mock_eval": {}},
            caught_exceptions=(ValueError,),
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
        pool.shutdown()

    def test_catches_timeout_exceptions(self):
        # this is a separate case because this exception
        # is raised at a slighly different location!
        pool = EvaluationPool(
            max_workers=4,
            evaluator_config={"mock_eval": {}},
            caught_exceptions=(TimeoutError,),
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
        pool.shutdown()

    def test_basic_caching(self):
        pool = EvaluationPool(
            max_workers=4,
            evaluator_config={"mock_eval": {}},
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
        pool.shutdown()

    def test_parallel_basic(self):
        # verify that something can happen in parallel
        pool = EvaluationPool(
            max_workers=20,
            evaluator_config={"mock_eval": {}},
        )

        times = 0.1*np.random.random(20)
        futures = [pool.schedule({"wait_for": t}) for t in times]

        @timed
        def wait_for_finished():
            return wait(futures)

        res, duration = wait_for_finished()
        done, undone = res

        for d in done:
            # make sure the results have been cached
            pool.finish(d)

        self.assertEqual(len(done), len(times))
        self.assertLess(duration, np.sum(times))  # did we achieve a speedup?
        self.assertEqual(len(pool.evals), len(times))
        pool.shutdown()

    def test_quadratic(self):
        # this is a somewhat redundant test, but it'll put the
        # pool through a slightly more realistic situation and hopefully
        # show us when and where it fails.

        space = {
            "x": ["hp_loggrid", "x", -2.0, 2.0, 161],
            "y": ["hp_loggrid", "y", -2.0, 2.0, 161],
            "z": ["hp_loggrid", "z", -2.0, 2.0, 161],
            "a": ["hp_loggrid", "a", -2.0, 2.0, 161],
            "b": ["hp_loggrid", "b", -2.0, 2.0, 161],
            "c": ["hp_loggrid", "c", -2.0, 2.0, 161],
            "wait": ["hp_choice", "wait", [0.0, 0.1]],
        }
        hpo = Hyperopt(space=space)

        pool = EvaluationPool(
            max_workers=40,
            evaluator_config={"mock_eval2": {}},
            trial_timeout=0.07,
        )

        search = Hyperopt(space=space)

        tasks = {}

        for i in range(10):
            t, s = hpo.suggest()
            tasks[t] = s

        futures = {pool.schedule(s): t for t, s in tasks.items()}

        for d in futures:
            pool.finish(d)
        pool.shutdown()

        self.assertEqual(len(pool.evals), 10)
