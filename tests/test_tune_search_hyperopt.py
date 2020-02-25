import numpy as np
from unittest import TestCase

from cmlkit.tune.search.hyperopt import Hyperopt


def target(d):
    return (d["x"] - 2.0) ** 2 + (d["y"] - 1.0) ** 2


def run_n_steps(opt, n, tape):
    for i in range(n):
        tid, s = opt.suggest()
        tape.append(("suggest", tid, s))
        res = target(s)
        opt.submit(tid, loss=res)
        tape.append(("submit", tid, res))

    return tape


def best_loss(tape):
    losses = []
    for entry in tape:
        if entry[0] == "submit":
            losses.append(entry[2])

    return min(losses)


def best_suggestion(tape):
    losses = []
    tids = []
    for entry in tape:
        if entry[0] == "submit":
            losses.append(entry[2])
            tids.append(entry[1])

    best = np.argmin(losses)
    best_tid = tids[best]

    for entry in tape:
        if entry[0] == "suggest":
            if entry[1] == best_tid:

                return entry


class TestHyperoptTPE(TestCase):
    def test_basic(self):
        space = {
            "x": ["hp_choice", "x", np.linspace(-10, 10, num=21)],
            "y": ["hp_choice", "y", np.linspace(-10, 10, num=21)],
        }

        hpo = Hyperopt(space=space, method="tpe")

        tape = []

        # does it do anything?
        tape = run_n_steps(hpo, 2, tape)
        start = best_loss(tape)
        tape = run_n_steps(hpo, 200, tape)
        end = best_loss(tape)
        self.assertGreater(start, end)

        # test repeatability
        tid, next_suggestion = hpo.suggest()

        hpo2 = Hyperopt(space=space, method="tpe")

        for entry in tape:
            if entry[0] == "suggest":
                tid, s = hpo2.suggest()
                self.assertEqual(tid, entry[1])
                self.assertEqual(s, entry[2])
            else:
                hpo2.submit(entry[1], loss=entry[2])

        tid, next_suggestion2 = hpo2.suggest()
        self.assertEqual(next_suggestion, next_suggestion2)

    def test_finds_min_choice(self):
        space = {
            "x": [
                "hp_choice",
                "x",
                np.linspace(-4, 4, num=41),
            ],  # reduce search space to make it easier
            "y": [
                "hp_choice",
                "y",
                np.linspace(-4, 4, num=41),
            ],  # reduce search space to make it easier
        }

        tape = []

        hpo = Hyperopt(space=space, method="tpe")
        run_n_steps(hpo, 50, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)

    def test_finds_min_uniform(self):
        space = {"x": ["hp_uniform", "x", -4.0, 4.0], "y": ["hp_uniform", "y", -4.0, 4.0]}

        tape = []

        hpo = Hyperopt(space=space, method="tpe")
        run_n_steps(hpo, 50, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)

    def test_finds_min_loggrid_uniform(self):
        space = {
            "x": ["hp_loggrid_uniform", "x", -2.0, 2.0, 0.05],
            "y": ["hp_loggrid_uniform", "y", -2.0, 2.0, 0.05],
        }

        tape = []

        hpo = Hyperopt(space=space, method="tpe")
        run_n_steps(hpo, 50, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)

    def test_finds_min_loggrid(self):
        space = {
            "x": ["hp_loggrid", "x", -2.0, 2.0, 81],
            "y": ["hp_loggrid", "y", -2.0, 2.0, 81],
        }

        tape = []

        hpo = Hyperopt(space=space, method="tpe")
        run_n_steps(hpo, 50, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)


class TestHyperoptTPE(TestCase):
    def test_basic(self):
        space = {
            "x": ["hp_choice", "x", np.linspace(-10, 10, num=21)],
            "y": ["hp_choice", "y", np.linspace(-10, 10, num=21)],
        }

        hpo = Hyperopt(space=space, method="rand")

        tape = []

        # does it do anything?
        tape = run_n_steps(hpo, 2, tape)
        start = best_loss(tape)
        tape = run_n_steps(hpo, 200, tape)
        end = best_loss(tape)
        self.assertGreater(start, end)

        # test repeatability
        tid, next_suggestion = hpo.suggest()

        hpo2 = Hyperopt(space=space, method="rand")

        for entry in tape:
            if entry[0] == "suggest":
                tid, s = hpo2.suggest()
                self.assertEqual(tid, entry[1])
                self.assertEqual(s, entry[2])
            else:
                hpo2.submit(entry[1], loss=entry[2])

        tid, next_suggestion2 = hpo2.suggest()
        self.assertEqual(next_suggestion, next_suggestion2)

    def test_finds_min_choice(self):
        space = {
            "x": [
                "hp_choice",
                "x",
                np.linspace(-4, 4, num=41),
            ],  # reduce search space to make it easier
            "y": [
                "hp_choice",
                "y",
                np.linspace(-4, 4, num=41),
            ],  # reduce search space to make it easier
        }

        tape = []

        hpo = Hyperopt(space=space, method="rand")
        run_n_steps(hpo, 200, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)

    def test_finds_min_uniform(self):
        space = {"x": ["hp_uniform", "x", -4.0, 4.0], "y": ["hp_uniform", "y", -4.0, 4.0]}

        tape = []

        hpo = Hyperopt(space=space, method="rand")
        run_n_steps(hpo, 200, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)

    def test_finds_min_loggrid_uniform(self):
        space = {
            "x": ["hp_loggrid_uniform", "x", -2.0, 2.0, 0.05],
            "y": ["hp_loggrid_uniform", "y", -2.0, 2.0, 0.05],
        }

        tape = []

        hpo = Hyperopt(space=space, method="rand")
        run_n_steps(hpo, 200, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)

    def test_finds_min_loggrid(self):
        space = {
            "x": ["hp_loggrid", "x", -2.0, 2.0, 81],
            "y": ["hp_loggrid", "y", -2.0, 2.0, 81],
        }

        tape = []

        hpo = Hyperopt(space=space, method="rand")
        run_n_steps(hpo, 200, tape)
        end = best_loss(tape)

        print(end)
        print(best_suggestion(tape))

        self.assertGreater(0.1, end)
