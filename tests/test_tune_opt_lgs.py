from unittest import TestCase
import numpy as np

from cmlkit.tune import OptimizerLGS


def f(x, y):
    return (x - 1.0) ** 2 + (y - 2.0) ** 2


class TestOptLGS(TestCase):
    def test_quadratic_minimum(self):
        lgs = OptimizerLGS()

        result = lgs(f, ((), ()))

        self.assertEqual(result["best"][0], 1.0)
        self.assertEqual(result["best"][1], 2.0)
