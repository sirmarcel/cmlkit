from unittest import TestCase

from cmlkit.tune.run.state import State
from cmlkit.tune.search.hyperopt import Hyperopt
from cmlkit.tune.run.stopping import StopMax


def target(d):
    return (d["x"] - 2.0) ** 2 + (d["y"] - 1.0) ** 2


class TestState(TestCase):
    def setUp(self):
        space = {
            "x": ["hp_loggrid_uniform", "x", -2.0, 2.0, 1.0],
            "y": ["hp_loggrid_uniform", "y", -2.0, 2.0, 1.0],
        }
        hpo = Hyperopt(space=space, method="tpe")
        state = State(search=hpo)

        results = []
        for i in range(50):
            tid, suggestion = state.suggest()
            results.append(
                (tid, {"ok": {"loss": target(suggestion), "suggestion": suggestion}})
            )

        for tid, result in results:
            state.submit(tid, result)

        self.state = state
        self.n_evals = len(state.evals)
        self.n_trials = len(state.trials)

    def test_stops_trials(self):
        stopper = StopMax(20, use="trials")

        self.assertTrue(stopper.done(self.state))

    def test_stops_trials(self):
        stopper = StopMax(20, use="trials")

        self.assertTrue(stopper.done(self.state))
        self.assertTrue(self.n_trials > 20)

    def test_stops_evals(self):
        # since there is no pool in the background
        # actually filling the evals cache, it is
        # empty and so the stopper will say: no!

        stopper = StopMax(20, use="evals")

        self.assertFalse(stopper.done(self.state))
        self.assertTrue(self.n_evals < 20)
