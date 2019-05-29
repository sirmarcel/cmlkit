import numpy as np
from unittest import TestCase

from cmlkit.tune2.run.resultdb import ResultDB


class TestResultDB(TestCase):
    def setUp(self):
        self.results_no_errors = {
            i: ["ok", {"loss": np.random.random(), "suggestion": {}}] for i in range(25)
        }

        self.states = [np.random.choice(["ok", "error"]) for i in range(25)]
        self.results_with_errors = {
            i: [state, {"loss": np.random.random(), "suggestion": {}}]
            for i, state in enumerate(self.states)
        }

    def test_basics_without_errors(self):
        # does instantiation do what we expect?
        resultdb = ResultDB()
        for k, v in self.results_no_errors.items():
            resultdb[k] = v

        self.assertEqual(resultdb.db, self.results_no_errors)

        # can we correctly obtain losses?
        tids, losses = resultdb.losses()
        real_losses = [v[1]["loss"] for v in self.results_no_errors.values()]

        self.assertEqual(losses, real_losses)
        self.assertEqual([resultdb.get_outcome(k)["loss"] for k in tids], real_losses)

        # can we obtain sorted losses?
        sorted_tids, sorted_losses = resultdb.sorted_losses()
        np.testing.assert_array_equal(sorted_losses, np.sort(real_losses))

        # are the keys also sorted?
        np.testing.assert_array_equal(
            [resultdb.get_outcome(k)["loss"] for k in sorted_tids], np.sort(real_losses)
        )

    def test_basics_with_errors(self):
        # does instantiation do what we expect?
        resultdb = ResultDB()
        for k, v in self.results_with_errors.items():
            resultdb[k] = v

        self.assertEqual(resultdb.db, self.results_with_errors)

        # can we correctly obtain losses, leaving out errors?
        tids, losses = resultdb.losses()
        real_losses = []
        for v in self.results_with_errors.values():
            if v[0] == "ok":
                real_losses.append(v[1]["loss"])

        self.assertEqual(len(losses), len(list(filter(lambda x: x == "ok", self.states))))
        self.assertEqual(losses, real_losses)

        error_counts = resultdb.count_by_error()

        self.assertEqual(
            error_counts["UnknownError"],
            len(list(filter(lambda x: x == "error", self.states))),
        )
