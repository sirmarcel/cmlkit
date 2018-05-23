from unittest import TestCase
from utils.hashing import *


class TestHashSpecDict(TestCase):

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

        digest = hash_spec_dict(d)

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

        digest1 = hash_spec_dict(d1)
        digest2 = hash_spec_dict(d2)

        self.assertNotEqual(digest1, digest2)
