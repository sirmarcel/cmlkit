from unittest import TestCase
import numpy as np
from cmlkit.optimize import *


class TestFindsMinimum(TestCase):

    def test_squared(self):
        def myf(x, y):
            return (x-1.0)**2.0 + (y-2.0)**2.0

        params = {'y': (-3.0, 3.0), 'x': (-5.0, 5.0)}

        result = optimize_bayes(myf, params)

        optimized_params = result['params']
        xy = np.array([optimized_params['x'], optimized_params['y']])

        expected = np.array([1.0, 2.0])

        np.testing.assert_allclose(xy, expected, atol=0.1)



class TestWrapFunctionWithNamedArgs(TestCase):

    def test_two_args(self):
        param_names = ['a', 'b']

        def myf(a, b):
            return a-b

        wrapped = wrap_function_with_named_args(myf, param_names)

        self.assertEqual(wrapped(np.array([3, 2])), 1)


    def test_three_args(self):
        param_names = ['a', 'b', 'c']

        def myf(a, b, c):
            return (a-b)*c

        wrapped = wrap_function_with_named_args(myf, param_names)

        self.assertEqual(wrapped(np.array([3, 2, 5])), 5)
