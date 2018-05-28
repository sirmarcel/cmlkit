import numpy as np
from unittest import TestCase
from unittest.mock import MagicMock
from hyperopt import hp
from qmmltools.autotune.parse import *
import qmmltools.stats as qmts


class TestHyperoptConversions(TestCase):
    # have to make sure the mocks get removed after the test,
    # otherwise they bleed over into other tests
    def setUp(self):
        self.og_choice = hp.choice

    def tearDown(self):
        hp.choice = self.og_choice

    def test_to_hyperopt(self):
        hp.choice = MagicMock()
        r = to_hyperopt(('hp_choice', 'mbtr_1', [1, 2, 3]))
        hp.choice.assert_called_with('mbtr_1', [1, 2, 3])

    def test_is_hyperopt(self):

        self.assertTrue(is_hyperopt(('hp_choice', 'mbtr_1', [1, 2, 3])))


class TestGridConversions(TestCase):

    def test_to_grid(self):
        r = to_grid(('gr_log2', -1, 1, 3))

        self.assertEqual(r.all(), np.array([0.5, 1.0, 2.0]).all())

    def test_is_grid(self):
        self.assertTrue(is_grid(('gr_log2', -1, 1, 3)))


class TestParsing(TestCase):
    # have to make sure the mocks get removed after the test,
    # otherwise they bleed over into other tests
    def setUp(self):
        self.og_choice = hp.choice
        self.og_rmse = qmts.rmse

    def tearDown(self):
        hp.choice = self.og_choice
        qmts.rmse = self.og_rmse

    def test_basic(self):
        d = {
            'one': {'mbtr1': ['hp_choice', 'mbtr_1', ['gr_log2', 0, 1, 1]],
                    'loss': 'rmse',
                    'other': 'something_else'},
            'two': 'heh'

        }

        hp.choice = MagicMock()
        qmts.rmse = 'hey_it_worked'

        parse(d)

        hp.choice.assert_called_with('mbtr_1', np.array([1.0]))
        self.assertEqual(d['one']['loss'], 'hey_it_worked')
