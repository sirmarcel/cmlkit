import numpy as np
from unittest import TestCase
from cmlkit.utils.hashing import *


class TestHashSortableDict(TestCase):

    def test_is_hash_persistent(self):
        d = {'mbtr_1': {'acc': 0.001,
                        'aindexf': 'full',
                        'corrf': 'identity',
                        'd': (-0.5, 0.5, 20),
                        'distrf': ('normal', 724.0773439350247),
                        'eindexf': 'full',
                        'elems': None,
                        'flatten': True,
                        'geomf': 'count',
                        'k': 1,
                        'norm': None,
                        'weightf': 'identity'},
             'mbtr_2': {'acc': 0.001,
                        'aindexf': 'noreversals',
                        'corrf': 'identity',
                        'd': (-0.04, 0.012, 100),
                        'distrf': ('normal', 0.3535533905932738),
                        'eindexf': 'noreversals',
                        'elems': None,
                        'flatten': True,
                        'geomf': '1/distance',
                        'k': 2,
                        'norm': None,
                        'weightf': ('exp_-1/identity', 6.06130303030303)},
             'mbtr_3': {'acc': 0.01,
                        'aindexf': 'noreversals',
                        'corrf': 'identity',
                        'd': (0, 0.105, 30),
                        'distrf': ('normal', 1.4142135623730951),
                        'eindexf': 'noreversals',
                        'elems': None,
                        'flatten': True,
                        'geomf': 'angle',
                        'k': 3,
                        'norm': None,
                        'weightf': '1/dotdotdot'}}

        digest = hash_sortable_dict(d)

        # Hash computed previously, should never change
        self.assertEqual(digest, '3a7b31e6ea64eaa5c627c3cfbb868914')

    def test_is_hash_different(self):
        d1 = {'mbtr_1': {'acc': 0.001,
                         'aindexf': 'full',
                         'corrf': 'identity',
                         'd': (-0.5, 0.5, 20),
                         'distrf': ('normal', 724.0773439350247),
                         'eindexf': 'full',
                         'elems': None,
                         'flatten': True,
                         'geomf': 'count',
                         'k': 1,
                         'norm': None,
                         'weightf': 'identity'},
              'mbtr_2': {'acc': 0.001,
                         'aindexf': 'noreversals',
                         'corrf': 'identity',
                         'd': (-0.04, 0.012, 100),
                         'distrf': ('normal', 0.3535533905932738),
                         'eindexf': 'noreversals',
                         'elems': None,
                         'flatten': True,
                         'geomf': '1/distance',
                         'k': 2,
                         'norm': None,
                         'weightf': ('exp_-1/identity', 6.06130303030303)},
              'mbtr_3': {'acc': 0.01,
                         'aindexf': 'noreversals',
                         'corrf': 'identity',
                         'd': (0, 0.105, 30),
                         'distrf': ('normal', 1.4142135623730951),
                         'eindexf': 'noreversals',
                         'elems': None,
                         'flatten': True,
                         'geomf': 'angle',
                         'k': 3,
                         'norm': None,
                         'weightf': '1/dotdotdot'}}

        d2 = {'mbtr_1': {'acc': 0.001,
                         'aindexf': 'full',
                         'corrf': 'identity',
                         'd': (-0.5, 0.5, 20),
                         'distrf': ('normal', 724.0773439350247),
                         'eindexf': 'full',
                         'elems': None,
                         'flatten': True,
                         'geomf': 'count',
                         'k': 1,
                         'norm': None,
                         'weightf': 'identity'},
              'mbtr_2': {'acc': 0.001,
                         'aindexf': 'noreversals',
                         'corrf': 'identity',
                         'd': (-0.04, 0.012, 100),
                         'distrf': ('normal', 0.3535533905932738),
                         'eindexf': 'noreversals',
                         'elems': None,
                         'flatten': True,
                         'geomf': '1/distance',
                         'k': 2,
                         'norm': None,
                         'weightf': ('exp_-1/identity', 6.08130303030303)},
              'mbtr_3': {'acc': 0.01,
                         'aindexf': 'noreversals',
                         'corrf': 'identity',
                         'd': (0, 0.105, 30),
                         'distrf': ('normal', 1.4142135623730951),
                         'eindexf': 'noreversals',
                         'elems': None,
                         'flatten': True,
                         'geomf': 'angle',
                         'k': 3,
                         'norm': None,
                         'weightf': '1/dotdotdot'}}

        digest1 = hash_sortable_dict(d1)
        digest2 = hash_sortable_dict(d2)

        self.assertNotEqual(digest1, digest2)


class TestHashArrays(TestCase):

    def setUp(self):
        self.a = np.array([[23.42, 1337.22], [12.12, 88.00]])
        self.b = None
        self.c = np.array([1, 2, 3], dtype=int)

    def test_stable_hash(self):
        h = hash_arrays(self.a, self.b, self.c)

        # Hash computed previously, should never change
        self.assertEqual(h, '4f5983bc6254a4c1a603c7ef464863db')

    def test_invariances(self):
        h1 = hash_arrays(np.array([[1, 2], [3, 4]]))
        h2 = hash_arrays(np.array([1, 2, 3, 4]))

        self.assertEqual(h1, h2)

        h1 = hash_arrays(np.array([[1, 2], [3, 4]]))
        h2 = hash_arrays(np.array([[3, 4], [1, 2]]))

        self.assertNotEqual(h1, h2)
