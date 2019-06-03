from unittest import TestCase
import random
import copy


from cmlkit.tune.run.state import State
from cmlkit.tune.search.hyperopt import Hyperopt


def target(d):
    return (d["x"] - 2.0) ** 2 + (d["y"] - 1.0) ** 2


class TestState(TestCase):
    def test_recording(self):
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

        random.shuffle(results)
        for tid, result in results:
            state.submit(tid, result)

        results = []
        for i in range(10):
            tid, suggestion = state.suggest()
            results.append(
                (tid, {"ok": {"loss": target(suggestion), "suggestion": suggestion}})
            )

        for i in range(10):
            tid, suggestion = state.suggest()

        random.shuffle(results)
        for tid, result in results:
            state.submit(tid, result)

        hpo2 = Hyperopt(space=space, method="tpe")
        state2 = State.from_tape(search=hpo2, tape=state.tape)

        for i in range(10):
            next_tid, next_suggestion = state.suggest()
            next_tid2, next_suggestion2 = state2.suggest()

            self.assertEqual(next_suggestion2, next_suggestion)
            self.assertEqual(next_tid2, next_tid)

        self.assertEqual(state2.tape, state.tape)
        self.assertEqual(state2.live_trials, state.live_trials)
        self.assertEqual(state2.trials.db, state.trials.db)

        # does the evals db do roughly what we expect it to? i.e. be
        # more unique than the trial one.
        self.assertGreater(len(state2.trials), len(state2.evals))

    def test_recording_fails_if_different(self):
        space = {"x": ["hp_uniform", "x", -4.0, 4.0], "y": ["hp_uniform", "y", -4.0, 4.0]}
        hpo = Hyperopt(space=space, method="tpe")
        state = State(search=hpo)

        results = []
        for i in range(50):
            tid, suggestion = state.suggest()
            results.append(
                (tid, {"ok": {"loss": target(suggestion), "suggestion": suggestion}})
            )

        random.shuffle(results)
        for tid, result in results:
            state.submit(tid, result)

        # different seed fails
        with self.assertRaises(AssertionError):
            hpo2 = Hyperopt(space=space, method="tpe", seed=1)
            state2 = State.from_tape(search=hpo2, tape=state.tape)

        # entry missing fails
        with self.assertRaises(AssertionError):
            tape = copy.copy(state.tape)
            del tape[21]
            hpo2 = Hyperopt(space=space, method="tpe")
            State.from_tape(search=hpo2, tape=tape)
