from unittest import TestCase
import shutil
import pathlib
import numpy as np
import time

import cmlkit
from cmlkit.engine import Component
from cmlkit.utility import timed

from cmlkit.tune.run import Run
from cmlkit.tune.search.hyperopt import Hyperopt


class MockEvaluator(Component):
    kind = "quadratic_eval"

    def __call__(self, model):
        time.sleep(model["wait"])
        loss = (
            (model["x"] - 2.0) ** 2
            + (model["y"] - 1.0) ** 2
            + (model["z"] - 0.0) ** 2
            + (model["a"] - 1.0) ** 2
            + (model["b"] - 2.0) ** 2
            + (model["c"] - 3.0) ** 2
        )
        return {"loss": loss}

    def _get_config(self):
        return {}


cmlkit.register(MockEvaluator)


class TestRun(TestCase):
    def setUp(self):
        self.tmpdir = pathlib.Path(__file__).parent / "tmp_test_run"
        self.tmpdir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_quadratic(self):
        space = {
            "x": ["hp_loggrid", "x", -2.0, 2.0, 161],
            "y": ["hp_loggrid", "y", -2.0, 2.0, 161],
            "z": ["hp_loggrid", "z", -2.0, 2.0, 161],
            "a": ["hp_loggrid", "a", -2.0, 2.0, 161],
            "b": ["hp_loggrid", "b", -2.0, 2.0, 161],
            "c": ["hp_loggrid", "c", -2.0, 2.0, 161],
            "wait": ["hp_choice", "wait", [0.0, 0.005, 0.01, 0.1]],
        }

        search = Hyperopt(space=space)

        run = Run(
            search=search,
            evaluator=MockEvaluator(),
            stop={"stop_max": {"count": 50}},
            context={"max_workers": 25},
            trial_timeout=0.08,
        )
        run.prepare(directory=self.tmpdir)
        run()

        run2 = Run.restore(
            directory=run.work_directory, new_stop={"stop_max": {"count": 100}}
        )
        run2.run()

        run3 = Run.checkout(directory=run.work_directory)

        self.assertEqual(run2.state.evals.db, run2.state.evals.db)
