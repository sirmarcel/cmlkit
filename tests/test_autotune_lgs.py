import numpy as np
from unittest import TestCase
from cmlkit.autotune.local_grid_search import *


class TestLGS(TestCase):
    def test_quadratic(self):
        d = {
            'a': '42',
            'b': {'par_1': 3.0, 'par_b': ['lgs', (5.0, 1, 1.0, -10, 10, -1, 2.0)], 'par_c': ['lgs', (5.0, 1, 1.0, -10, 10, -1, 2.0)]}
        }


        def target(args):
            b = args['b']

            return b['par_1'] + (b['par_b'] - 0.5)**2 + + (b['par_c'] - 0.125)**2


        res = run_lgs(d, target)

        self.assertEqual(d['b']['par_b'], 0.5)
        self.assertEqual(d['b']['par_c'], 0.125)
